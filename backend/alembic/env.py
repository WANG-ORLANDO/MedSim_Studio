"""
Alembic migrations environment configuration.

Loads all SQLAlchemy models so that --autogenerate can detect schema changes.
"""

import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Alembic Config object (provides access to alembic.ini values)
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so Alembic autogenerate detects their tables
# -----------------------------------------------------------------------
# SQLAlchemy recommends importing the declarative Base here and setting
# target_metadata = Base.metadata so that --autogenerate can diff the
# current schema against the actual database.
from app.database.session import Base
import app.models  # noqa: F401 — registers all ORM models with Base.metadata

target_metadata = Base.metadata

# Other values from alembic.ini, or set programmatically here
# e.g. config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def filter_unneeded_metadata(metadata, tables_to_exclude=None):
    """
    Remove tables from metadata that should not be managed by Alembic.
    Useful for excluding SQLAlchemy internal tables or third-party tables.
    """
    if tables_to_exclude is None:
        tables_to_exclude = set()
    for table_name in list(metadata.tables.keys()):
        if table_name in tables_to_exclude:
            del metadata.tables[table_name]
    return metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine.
    Calls to context.execute() here emit the SQL to a script file.
    This is useful for generating SQL migration scripts without
    connecting to a live database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine from the alembic.ini URL and associates it
    with the Alembic context so migration operations execute against
    a live database connection.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
