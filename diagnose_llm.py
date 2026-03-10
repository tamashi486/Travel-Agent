"""
diagnose_llm.py — LLM API 健康诊断
========================================
测试三件事：
  1. 原始延迟：直接调 LLM API，绕开所有中间层
  2. 是否限流：检查响应头和错误类型
  3. 响应一致性：连发 3 次，看延迟是否稳定

用法：
  conda run -n travel python diagnose_llm.py
"""

import asyncio
import time
import sys
from pathlib import Path

# 加载项目 .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)

from openai import AsyncOpenAI
import os

BASE_URL = os.getenv("LLM_BASE_URL", "")
API_KEY  = os.getenv("LLM_API_KEY", "")
MODEL    = os.getenv("LLM_MODEL_ID", "gpt-4o")

if not API_KEY:
    print("❌ LLM_API_KEY 未配置，请检查 .env")
    sys.exit(1)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL or None, timeout=60)

PROMPT_SHORT = "回复'ok'"
PROMPT_LONG  = "用中文写一段50字的北京旅游介绍"


async def single_call(prompt: str, label: str) -> dict:
    """发一次请求，返回延迟和结果摘要"""
    start = time.time()
    error = None
    status = None
    reply = ""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        elapsed = (time.time() - start) * 1000
        reply = resp.choices[0].message.content[:60] if resp.choices else ""
        status = "ok"
        # 检查是否有速率限制相关头（部分 API 会返回）
        headers = getattr(resp, "_raw_response", None)
        rl_remaining = None
        if headers:
            rl_remaining = headers.headers.get("x-ratelimit-remaining-requests")
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        error = str(e)
        status = "error"
        rl_remaining = None

    result = {
        "label": label,
        "latency_ms": round(elapsed),
        "status": status,
        "reply": reply,
        "error": error,
        "rl_remaining": rl_remaining,
    }
    return result


async def main():
    print(f"\n{'='*55}")
    print(f"  LLM API 诊断")
    print(f"  Model:    {MODEL}")
    print(f"  Base URL: {BASE_URL or '(OpenAI 默认)'}")
    print(f"{'='*55}\n")

    results = []

    # ── 测试1: 单次短请求延迟 ──────────────────────────────
    print("▶ [1/3] 单次短请求 (max_tokens=100)...")
    r = await single_call(PROMPT_SHORT, "短请求")
    results.append(r)
    _print_result(r)

    # ── 测试2: 连发3次，看一致性 ──────────────────────────
    print("\n▶ [2/3] 连发 3 次短请求，检查延迟稳定性...")
    burst = []
    for i in range(3):
        r = await single_call(PROMPT_SHORT, f"连发#{i+1}")
        burst.append(r)
        _print_result(r)
        await asyncio.sleep(0.5)

    latencies = [r["latency_ms"] for r in burst if r["status"] == "ok"]
    if len(latencies) >= 2:
        spread = max(latencies) - min(latencies)
        print(f"  → 延迟范围: {min(latencies)}ms ~ {max(latencies)}ms  (抖动: {spread}ms)")
        if spread > 10000:
            print("  ⚠️  延迟抖动超过 10s，可能存在限流队列或服务不稳定")

    # ── 测试3: 较长请求 ───────────────────────────────────
    print(f"\n▶ [3/3] 较长请求 (生成旅游介绍)...")
    r = await single_call(PROMPT_LONG, "长请求")
    results.append(r)
    _print_result(r)

    # ── 综合判断 ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  综合诊断结论")
    print(f"{'='*55}")

    ok_results = [r for r in results + burst if r["status"] == "ok"]
    err_results = [r for r in results + burst if r["status"] == "error"]

    if err_results:
        print(f"❌ {len(err_results)} 次请求失败:")
        for r in err_results:
            err_lower = r["error"].lower()
            if "429" in r["error"] or "rate" in err_lower:
                print(f"   → {r['label']}: 触发限流 (429) — {r['error'][:80]}")
            elif "timeout" in err_lower or "timed out" in err_lower:
                print(f"   → {r['label']}: 请求超时 — 可能 API 拥堵")
            elif "connection" in err_lower:
                print(f"   → {r['label']}: 连接失败 — 检查网络/BASE_URL 配置")
            else:
                print(f"   → {r['label']}: {r['error'][:80]}")

    if ok_results:
        avg_ms = sum(r["latency_ms"] for r in ok_results) / len(ok_results)
        print(f"\n✅ 成功请求: {len(ok_results)} 次，平均延迟: {avg_ms:.0f}ms")
        if avg_ms < 3000:
            print("   → API 状态正常，延迟良好")
        elif avg_ms < 15000:
            print("   → API 略慢，可能轻度拥堵，但可用")
        elif avg_ms < 60000:
            print("   → API 明显变慢（10-60s），存在拥堵或限流排队")
        else:
            print("   → API 极度缓慢（>60s），强烈建议等待后重试 benchmark")

    if not err_results and ok_results:
        print("\n  无 429/限流错误，慢速原因更可能是：")
        print("  • DeepSeek API 服务器当前负载高")
        print("  • 账号触发软限流（请求排队而非直接拒绝）")
        print("  • 网络链路延迟（如代理/VPN 路由问题）")


def _print_result(r: dict):
    status_icon = "✅" if r["status"] == "ok" else "❌"
    extra = f"  reply: {r['reply']!r}" if r["reply"] else f"  error: {r['error'][:80]}"
    rl = f"  rl_remaining={r['rl_remaining']}" if r["rl_remaining"] else ""
    print(f"  {status_icon} {r['label']}: {r['latency_ms']}ms{rl}")
    print(f"     {extra}")


if __name__ == "__main__":
    asyncio.run(main())
