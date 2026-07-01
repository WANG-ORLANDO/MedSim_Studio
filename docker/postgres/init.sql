-- ============================================================
-- CT Simulator - PostgreSQL Initialization Script
-- ============================================================
-- Creates initial database schema for the CT Simulator platform.
-- Tables are created by SQLAlchemy on first startup, but this
-- script provides the initial schema for Docker deployments.
-- ============================================================

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Verify database connection
DO $$
BEGIN
    RAISE NOTICE 'CT Simulator database initialized successfully';
END $$;
