# API Overview

## Base URL

```
Development: http://localhost:8000/api/v1
Production:  https://your-domain.com/api/v1
```

## Authentication

All API endpoints (except health checks) require authentication via JWT bearer token.

## Common Headers

- `Content-Type: application/json`
- `Authorization: Bearer <token>`

## Error Format

```json
{
  "detail": "Error message description"
}
```

## Rate Limiting

- 100 requests/minute per IP for unauthenticated users
- 1000 requests/minute for authenticated users
