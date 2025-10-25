from sqlalchemy import Column, Integer, Text, Boolean, String, TIMESTAMP
from sqlalchemy.dialects.mysql import VARCHAR, TINYINT, BIGINT
from database import Base  # assuming you have Base defined

class VideoViolation(Base):
    __tablename__ = "attachments"

    id = Column(String(36), primary_key=True)  # char(36)
    filename = Column(Text, nullable=False)
    violation_detected = Column(TINYINT(1), nullable=False)
    driver = Column(Integer, nullable=False)
    type = Column(VARCHAR(191), nullable=False, index=True)
    user_id = Column(BIGINT(unsigned=True), nullable=True, index=True)
    post_id = Column(BIGINT(unsigned=True), nullable=True, index=True)
    story_id = Column(BIGINT(unsigned=True), nullable=True, index=True)
    message_id = Column(BIGINT(unsigned=True), nullable=True, index=True)
    collab_id = Column(BIGINT(unsigned=True), nullable=True, index=True)
    coconut_id = Column(VARCHAR(191), nullable=True, index=True)
    has_thumbnail = Column(TINYINT(1), nullable=True)
    has_blurred_preview = Column(TINYINT(1), nullable=True)
    created_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=True)
    payment_request_id = Column(BIGINT(unsigned=True), nullable=True, index=True)
