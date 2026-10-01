"""VoiceShield V2 Initial PostgreSQL Baseline Schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-10-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

json_type = sa.JSON().with_variant(postgresql.JSONB, "postgresql")


def upgrade() -> None:
    # 1. analysis_records
    op.create_table(
        'analysis_records',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('stored_audio_path', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('classification', sa.String(length=32), nullable=False),
        sa.Column('risk_score', sa.Integer(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('ai_probability', sa.Float(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=False),
        sa.Column('segments_analyzed', sa.Integer(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('detector_version', sa.String(length=64), nullable=False),
        sa.Column('reasons', json_type, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_records_id'), 'analysis_records', ['id'], unique=False)

    # 2. call_sessions
    op.create_table(
        'call_sessions',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('client_session_id', sa.String(length=128), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('current_risk', sa.String(length=32), nullable=False),
        sa.Column('peak_risk', sa.String(length=32), nullable=False),
        sa.Column('current_score', sa.Float(), nullable=False),
        sa.Column('peak_score', sa.Float(), nullable=False),
        sa.Column('realtime_alert_state', sa.String(length=32), nullable=False),
        sa.Column('final_risk', sa.String(length=32), nullable=True),
        sa.Column('final_score', sa.Float(), nullable=True),
        sa.Column('final_provider', sa.String(length=64), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('last_notification_at', sa.DateTime(), nullable=True),
        sa.Column('final_audio_reference', sa.String(length=255), nullable=True),
        sa.Column('final_classification', sa.String(length=64), server_default='ANALYZING', nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('chunks_analyzed', sa.Integer(), server_default='0', nullable=True),
        sa.Column('total_duration_seconds', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('evidence', json_type, nullable=True),
        sa.Column('detector_version', sa.String(length=64), server_default='aurigin-realtime', nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('client_session_id', name='uq_call_sessions_client_session_id')
    )
    op.create_index(op.f('ix_call_sessions_id'), 'call_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_call_sessions_client_session_id'), 'call_sessions', ['client_session_id'], unique=True)

    # 3. call_chunks
    op.create_table(
        'call_chunks',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('call_session_id', sa.String(length=64), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(length=64), nullable=False),
        sa.Column('provider_prediction_id', sa.String(length=128), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('risk', sa.String(length=32), nullable=True),
        sa.Column('processing_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('idempotency_key', sa.String(length=128), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['call_session_id'], ['call_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('call_session_id', 'sequence_number', name='uq_call_chunk_session_sequence'),
        sa.UniqueConstraint('call_session_id', 'idempotency_key', name='uq_call_chunk_session_idempotency_key')
    )
    op.create_index(op.f('ix_call_chunks_id'), 'call_chunks', ['id'], unique=False)
    op.create_index(op.f('ix_call_chunks_call_session_id'), 'call_chunks', ['call_session_id'], unique=False)
    op.create_index(op.f('ix_call_chunks_idempotency_key'), 'call_chunks', ['idempotency_key'], unique=False)

    # 4. device_recording_compatibilities
    op.create_table(
        'device_recording_compatibilities',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('manufacturer', sa.String(length=64), nullable=False),
        sa.Column('model', sa.String(length=128), nullable=False),
        sa.Column('os_family', sa.String(length=64), nullable=True),
        sa.Column('os_version', sa.String(length=64), nullable=True),
        sa.Column('android_version', sa.String(length=64), nullable=True),
        sa.Column('recording_profile', sa.String(length=32), nullable=False),
        sa.Column('supported', sa.Boolean(), nullable=False),
        sa.Column('configuration', json_type, nullable=False),
        sa.Column('profile_version', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('marketing_name', sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('manufacturer', 'model', name='uq_device_recording_compatibility_mfg_model')
    )
    op.create_index(op.f('ix_device_recording_compatibilities_id'), 'device_recording_compatibilities', ['id'], unique=False)
    op.create_index(op.f('ix_device_recording_compatibilities_manufacturer'), 'device_recording_compatibilities', ['manufacturer'], unique=False)
    op.create_index(op.f('ix_device_recording_compatibilities_model'), 'device_recording_compatibilities', ['model'], unique=False)

    # 5. users
    op.create_table(
        'users',
        sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('display_name', sa.String(length=128), nullable=True),
        sa.Column('firebase_uid', sa.String(length=128), nullable=False),
        sa.Column('status', sa.String(length=32), server_default='ACTIVE', nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('phone_number', name='uq_users_phone_number'),
        sa.UniqueConstraint('firebase_uid', name='uq_users_firebase_uid')
    )
    op.create_index(op.f('ix_users_phone_number'), 'users', ['phone_number'], unique=True)
    op.create_index(op.f('ix_users_firebase_uid'), 'users', ['firebase_uid'], unique=True)

    # 6. user_devices
    op.create_table(
        'user_devices',
        sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('device_fingerprint', sa.String(length=255), nullable=False),
        sa.Column('manufacturer', sa.String(length=128), nullable=True),
        sa.Column('model', sa.String(length=128), nullable=True),
        sa.Column('android_version', sa.String(length=64), nullable=True),
        sa.Column('app_version', sa.String(length=64), nullable=True),
        sa.Column('first_seen_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('device_fingerprint', name='uq_user_devices_device_fingerprint')
    )
    op.create_index(op.f('ix_user_devices_user_id'), 'user_devices', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_devices_device_fingerprint'), 'user_devices', ['device_fingerprint'], unique=True)

    # 7. user_auth_events
    op.create_table(
        'user_auth_events',
        sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('firebase_uid', sa.String(length=128), nullable=True),
        sa.Column('phone_number', sa.String(length=20), nullable=True),
        sa.Column('event_type', sa.String(length=32), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('device_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['device_id'], ['user_devices.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_auth_events_user_id'), 'user_auth_events', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_auth_events_firebase_uid'), 'user_auth_events', ['firebase_uid'], unique=False)
    op.create_index(op.f('ix_user_auth_events_device_id'), 'user_auth_events', ['device_id'], unique=False)

    # 8. cybercrime_categories
    op.create_table(
        'cybercrime_categories',
        sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', name='uq_cybercrime_categories_name')
    )
    op.create_index(op.f('ix_cybercrime_categories_name'), 'cybercrime_categories', ['name'], unique=True)

    # 9. cybercrime_templates
    op.create_table(
        'cybercrime_templates',
        sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('category_id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('severity', sa.String(length=32), nullable=True),
        sa.Column('indicators', json_type, server_default=sa.text("'[]'::jsonb"), nullable=True),
        sa.Column('recommended_actions', json_type, server_default=sa.text("'[]'::jsonb"), nullable=True),
        sa.Column('metadata', json_type, server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['category_id'], ['cybercrime_categories.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('category_id', 'name', name='uq_cybercrime_template_category_name')
    )
    op.create_index(op.f('ix_cybercrime_templates_category_id'), 'cybercrime_templates', ['category_id'], unique=False)

    # 10. cybercrime_template_versions
    op.create_table(
        'cybercrime_template_versions',
        sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('template_id', sa.Uuid(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('template_data', json_type, nullable=False),
        sa.Column('change_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['template_id'], ['cybercrime_templates.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('template_id', 'version', name='uq_cybercrime_template_version')
    )
    op.create_index(op.f('ix_cybercrime_template_versions_template_id'), 'cybercrime_template_versions', ['template_id'], unique=False)


def downgrade() -> None:
    op.drop_table('cybercrime_template_versions')
    op.drop_table('cybercrime_templates')
    op.drop_table('cybercrime_categories')
    op.drop_table('user_auth_events')
    op.drop_table('user_devices')
    op.drop_table('users')
    op.drop_table('device_recording_compatibilities')
    op.drop_table('call_chunks')
    op.drop_table('call_sessions')
    op.drop_table('analysis_records')
