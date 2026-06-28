---
name: yolo-api-data-layer
description: Use this skill for YOLO FastAPI data-layer tasks in services/yolo, especially refactoring from raw sqlite3 to SQLAlchemy ORM, creating models, updating database-backed endpoints, supporting SQLite/PostgreSQL, and writing tests with temporary SQLite and mocked YOLO inference.
---

# YOLO API Data Layer Skills

1. Use SQLAlchemy ORM instead of raw sqlite3.
2. Do not use raw SQL strings.
3. Do not use conn.execute or cursor.execute.
4. Create a separate services/yolo/db.py file.
5. Define Base, engine, SessionLocal, and get_db in db.py.
6. Define PredictionSession as a SQLAlchemy model.
7. Define DetectionObject as a SQLAlchemy model.
8. Use Base.metadata.create_all for database initialization.
9. Read DATABASE_URL from environment variables.
10. Default to SQLite for local development.
11. Support PostgreSQL through DATABASE_URL.
12. Use FastAPI Depends(get_db) in DB endpoints.
13. Use SQLAlchemy Session for all DB reads and writes.
14. Replace INSERT logic with db.add and db.commit.
15. Replace SELECT logic with db.query.
16. Preserve all existing endpoints.
17. Preserve all status codes.
18. Preserve all response JSON structures.
19. Use temporary SQLite databases in tests.
20. Mock the YOLO model in API tests.
21. Create services/yolo/models.py for SQLAlchemy ORM models, and keep database configuration such as engine, SessionLocal, and get_db in services/yolo/db.py.
