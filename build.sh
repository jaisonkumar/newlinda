#!/bin/bash
set -e

# install dependencies (Render already runs pip install but safe to include)
pip install -r requirements.txt

# Run migrations and collect static
python manage.py migrate --noinput
python manage.py collectstatic --noinput
