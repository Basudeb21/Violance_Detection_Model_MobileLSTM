from datetime import datetime
import uuid
from database import get_db
from model import VideoViolation

def save_video_violation_to_db(data: dict, violation: int):
    """
    Save video violation info into DB.
    data: dict with keys like filename, driver, type, user_id, post_id, etc.
    violation: 1 if violation detected, 0 otherwise
    """
    db_gen = get_db()
    db = next(db_gen)

    try:
        new_violation = VideoViolation(
            id=str(uuid.uuid4()),
            filename=data.get("filename"),
            violation_detected=violation,
            driver=data.get("driver", 1),
            type=data.get("type", "video"),
            user_id=data.get("user_id"),
            post_id=data.get("post_id"),
            story_id=data.get("story_id"),
            message_id=data.get("message_id"),
            collab_id=data.get("collab_id"),
            coconut_id=data.get("coconut_id"),
            has_thumbnail=data.get("has_thumbnail", 0),
            has_blurred_preview=data.get("has_blurred_preview", 0),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            payment_request_id=data.get("payment_request_id")
        )
        db.add(new_violation)
        db.commit()
        db.refresh(new_violation)
        print(f"✅ Video violation saved: ID {new_violation.id}")
    except Exception as e:
        db.rollback()
        print("❌ DB Save Error:", repr(e))
    finally:
        db.close()
