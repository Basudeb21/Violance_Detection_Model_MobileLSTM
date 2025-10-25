from database import get_db
from sqlalchemy import text

def test_connection():
    # Get a session from your generator
    db_gen = get_db()
    db = next(db_gen)
    try:
        # Execute a simple query
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            print("✅ Database connection successful!")
        else:
            print("❌ Database connected but test query failed.")
    except Exception as e:
        print("❌ Database connection failed:", e)
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()
