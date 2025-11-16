# checkpoint_viewer.py
import pickle
from pathlib import Path


def view_checkpoint(file_path: str):
    """Checkpoint 파일 내용 확인"""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        print(f"파일: {file_path}")
        print(f"타입: {type(data)}")
        print(f"\n=== 전체 구조 ===")

        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n[{key}]")
                print(f"  타입: {type(value)}")

                # 주요 데이터 미리보기
                if key == "channel_values" and isinstance(value, dict):
                    for ch_key, ch_value in value.items():
                        print(f"    {ch_key}: {type(ch_value)}")
                        if ch_key == "messages" and isinstance(ch_value, list):
                            print(f"      메시지 개수: {len(ch_value)}")
                            for i, msg in enumerate(ch_value[:3]):  # 처음 3개만
                                print(f"\n      Message {i}:")
                                print(f"        타입: {type(msg).__name__}")
                                if hasattr(msg, "content"):
                                    content = str(msg.content)[:100]
                                    print(f"        내용: {content}...")
                elif isinstance(value, (str, int, float, bool)):
                    print(f"  값: {value}")
        else:
            print(data)

        return data
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def list_checkpoint_files():
    """langgraph_api 디렉토리의 모든 checkpoint 파일 나열"""
    api_dir = Path(".langgraph_api")

    if not api_dir.exists():
        print("❌ .langgraph_api 디렉토리가 없습니다.")
        print("💡 LangGraph 서버를 먼저 실행해주세요: langgraph dev")
        return []

    files = list(api_dir.glob("*.pckl"))

    if not files:
        print("❌ checkpoint 파일이 없습니다.")
        return []

    print("📁 발견된 checkpoint 파일:")
    for f in sorted(files):
        size = f.stat().st_size
        print(f"  - {f.name} ({size:,} bytes)")

    return files


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph Checkpoint Viewer")
    print("=" * 60)

    # 1. 파일 목록 확인
    files = list_checkpoint_files()

    if not files:
        exit(1)

    print("\n" + "=" * 60)

    # 2. 첫 번째 checkpoint 파일 분석
    checkpoint_file = files[0]
    print(f"\n분석 대상: {checkpoint_file}")
    print("=" * 60)

    data = view_checkpoint(str(checkpoint_file))

    # 3. store.pckl이 있으면 확인
    store_file = Path(".langgraph_api/store.pckl")
    if store_file.exists():
        print("\n" + "=" * 60)
        print("Store 파일 분석")
        print("=" * 60)
        view_checkpoint(str(store_file))
