"""验证本地 tmp 教学脚本的流读取；干净检出不包含 tmp 时跳过这些测试。"""

import asyncio
import importlib.util
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import httpx
import pytest


@pytest.fixture
def trace_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """导入本地教学脚本；tmp 不属于正式项目交付，不强制要求它随 git 存在。"""
    path = Path(__file__).resolve().parents[2] / "tmp/trace_sse.py"
    if not path.exists():
        pytest.skip("本地 tmp/trace_sse.py 不存在")
    spec = importlib.util.spec_from_file_location("trace_sse_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class ChunkedStream(httpx.AsyncByteStream):
    """每次只交付 7 字节，确保行、JSON 和中文字符会跨网络块。"""

    def __init__(self, wire: bytes, delay: float = 0.0) -> None:
        self.wire = wire
        self.delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self.wire), 7):
            await asyncio.sleep(self.delay)
            yield self.wire[offset : offset + 7]


async def run_trace(module: ModuleType, wire: str, delay: float = 0.0) -> object:
    """通过内存 HTTP 流运行真实解析循环，测试不会请求 GPU。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            stream=ChunkedStream(wire.encode(), delay),
        )

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        return await module.trace_stream(client, "demo", "hello", 8, 0.05)


def test_fragmented_sse_explains_boundaries_and_usage(
    trace_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """角色、跨行 JSON、文本、结束原因、空 choices usage 和 DONE 都应被识别。"""
    wire = (
        ": heartbeat\n\n"
        "event: message\n"
        'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        'data: {"choices":\n'
        'data: [{"delta":{"content":"你好"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    )
    result = asyncio.run(run_trace(trace_module, wire))
    assert result.output_text == "你好"
    assert result.text_event_count == 1
    assert result.completion_tokens == 3
    assert result.event_count == 5
    assert result.saw_done is True
    output = capsys.readouterr().out
    assert "SSE 注释/心跳" in output
    assert "角色声明" in output
    assert "finish_reason='stop'" in output
    assert "choices 数量：0" in output
    assert "[LINE" in output and "空行" in output


@pytest.mark.parametrize(
    "wire",
    [
        'data: {"choices":[]}\n\n',  # 没有 DONE
        "data: [DONE]\n",  # 没有事件结束空行
        'data: {"error":{"message":"failed"}}\n\n',
        "data: []\n\n",
        "data: {broken JSON}\n\n",
        'data: {"choices":[],"usage":{"completion_tokens":true}}\n\n',
    ],
)
def test_invalid_or_incomplete_stream_fails(
    trace_module: ModuleType, wire: str
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(run_trace(trace_module, wire))


def test_whole_request_timeout(trace_module: ModuleType) -> None:
    with pytest.raises(TimeoutError):
        asyncio.run(run_trace(trace_module, "data: [DONE]\n\n", delay=0.2))
