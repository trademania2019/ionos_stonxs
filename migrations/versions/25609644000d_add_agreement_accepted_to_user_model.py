"""Add agreement_accepted to User model

Revision ID: 25609644000d
Revises: 
Create Date: 2024-07-05 01:12:38.824000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '25609644000d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add the 'agreement_accepted' column to the 'users' table
    op.add_column('users', sa.Column('agreement_accepted', sa.Boolean(), nullable=True))


def downgrade():
    # Remove the 'agreement_accepted' column from the 'users' table
    op.drop_column('users', 'agreement_accepted')
