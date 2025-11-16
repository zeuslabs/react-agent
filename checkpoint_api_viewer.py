# checkpoint_api_viewer.py (수정됨)
import asyncio
from langgraph_sdk import get_client


async def view_threads():
    """LangGraph API를 통해 threads 조회"""

    # 로컬 서버에 연결
    async with get_client(url="http://localhost:2024") as client:

        # 1. 모든 assistants 확인
        print("📋 Available Assistants:")
        assistants = await client.assistants.search()
        for assistant in assistants:
            print(f"  - {assistant['assistant_id']}")

        if not assistants:
            print("❌ Assistant가 없습니다.")
            return

        assistant_id = assistants[0]["assistant_id"]
        print(f"\n🎯 Using assistant: {assistant_id}")

        # 2. Threads 목록 가져오기
        print("\n📝 Threads:")
        threads = await client.threads.search()

        if not threads:
            print("  (대화 기록 없음)")
            return

        for thread in threads:
            thread_id = thread["thread_id"]
            print(f"\n  {'='*50}")
            print(f"  Thread ID: {thread_id}")
            print(f"  Created: {thread.get('created_at', 'N/A')}")
            print(f"  Updated: {thread.get('updated_at', 'N/A')}")

            try:
                # 3. Thread의 상태 가져오기 (수정된 방법)
                state = await client.threads.get_state(
                    thread_id=thread_id, subgraphs=True  # 서브그래프 포함
                )

                print(f"  \n  📊 State Info:")
                print(f"    - Values keys: {list(state.get('values', {}).keys())}")

                # Messages 확인
                if "values" in state:
                    messages = state["values"].get("messages", [])
                    print(f"    - Total messages: {len(messages)}")

                    if messages:
                        print(f"\n  💬 Recent Messages (last 5):")
                        for i, msg in enumerate(messages[-5:], 1):
                            msg_type = msg.get("type", "unknown")

                            # Content 추출
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                # content가 리스트인 경우 (multimodal)
                                text_parts = [
                                    item.get("text", "")
                                    for item in content
                                    if isinstance(item, dict)
                                    and item.get("type") == "text"
                                ]
                                content = " ".join(text_parts)

                            content_preview = str(content)[:100]
                            print(f"    {i}. [{msg_type}] {content_preview}...")

                # 4. Thread history (checkpoints) 확인
                print(f"\n  📜 Checkpoint History:")
                history = await client.threads.get_history(
                    thread_id=thread_id, limit=5  # 최근 5개만
                )

                for idx, checkpoint in enumerate(history, 1):
                    checkpoint_id = checkpoint.get("checkpoint_id", "N/A")
                    parent_id = checkpoint.get("parent_checkpoint_id", "None")
                    print(f"    {idx}. Checkpoint: {checkpoint_id[:8]}...")
                    print(f"       Parent: {parent_id[:8] if parent_id else 'None'}...")

                    # Checkpoint의 values 확인
                    if "values" in checkpoint:
                        msg_count = len(checkpoint["values"].get("messages", []))
                        print(f"       Messages: {msg_count}")

            except Exception as e:
                print(f"  ❌ 상태 조회 실패: {e}")
                continue


async def view_single_thread(thread_id: str):
    """특정 thread의 상세 정보 조회"""
    async with get_client(url="http://localhost:2024") as client:

        print(f"\n🔍 Thread {thread_id} 상세 정보")
        print("=" * 60)

        try:
            # Thread 정보
            thread_info = await client.threads.get(thread_id)
            print(f"Created: {thread_info.get('created_at')}")
            print(f"Updated: {thread_info.get('updated_at')}")

            # State
            state = await client.threads.get_state(thread_id)
            messages = state.get("values", {}).get("messages", [])

            print(f"\n💬 전체 대화 ({len(messages)} messages):")
            print("=" * 60)

            for i, msg in enumerate(messages, 1):
                msg_type = msg.get("type", "unknown")
                msg_id = msg.get("id", "N/A")

                # Content 추출
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)

                print(f"\n[{i}] {msg_type.upper()} (ID: {msg_id[:8]}...)")
                print("-" * 60)
                print(content)

                # Tool calls 확인
                if msg.get("tool_calls"):
                    print("\n  🔧 Tool Calls:")
                    for tc in msg["tool_calls"]:
                        print(f"    - {tc.get('name')}: {tc.get('args')}")

        except Exception as e:
            print(f"❌ 오류: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph API Viewer")
    print("=" * 60)
    print("\n⚠️  먼저 LangGraph 서버를 실행해주세요:")
    print("    langgraph dev\n")
    print("=" * 60)

    import sys

    try:
        if len(sys.argv) > 1:
            # 특정 thread ID가 주어진 경우
            thread_id = sys.argv[1]
            asyncio.run(view_single_thread(thread_id))
        else:
            # 전체 threads 조회
            asyncio.run(view_threads())

    except KeyboardInterrupt:
        print("\n\n중단됨")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        print("\n💡 LangGraph 서버가 실행 중인지 확인해주세요:")
        print("    langgraph dev")
