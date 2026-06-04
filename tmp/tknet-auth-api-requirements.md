# tknet.ai Authentication & Model Catalog API — Technical Requirements

## Overview

NovaMLX (macOS AI inference app) currently authenticates against `novamlx.ai` and fetches managed model lists from there. We are migrating the auth and model catalog backend to `tknet.ai`. This document specifies the exact API contracts that `tknet.ai` must implement for NovaMLX to work.

**Base URL**: `https://tknet.ai`

---

## 1. Login API

### `POST /api/v1/auth/login`

Authenticates a user with email and password. Returns a session token and user profile.

#### Request

```json
{
  "email": "user@example.com",
  "password": "plaintext-password"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string | ✅ | User's email address |
| password | string | ✅ | User's password (plaintext over HTTPS) |

#### Response — 200 OK

```json
{
  "session": "string-session-token",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "name": "User Name",
    "plan": "pro"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| session | string | Session token. Must be a non-guessable string (JWT or opaque token). Used for all subsequent authenticated requests. Should remain valid for at least 7 days. |
| user.id | integer | Internal user ID |
| user.email | string | User's email |
| user.name | string \| null | Display name (optional) |
| user.plan | string | Subscription plan. One of: `"free"`, `"pro"`, `"enterprise"`. `"pro"` or above = subscribed user with unlimited provider access. |

#### Error Responses

| Status | Meaning | Body |
|--------|---------|------|
| 401 | Invalid email or password | Any (client checks status code only) |
| 429 | Rate limited (too many attempts) | Any (client checks status code only) |
| 5xx | Server error | Any |

---

## 2. Session Check API

### `POST /api/v1/auth/check`

Validates an existing session token and returns the user's current subscription status. This is called periodically (every ~5 minutes) by the client to re-validate.

#### Request

```json
{
  "session": "string-session-token"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| session | string | ✅ | The session token obtained from `/login` |

#### Response — 200 OK (subscribed user)

```json
{
  "valid": true,
  "plan": "pro",
  "status": "active",
  "cancel_at_period_end": false,
  "expires_at": "2026-12-31T23:59:59Z",
  "user": {
    "email": "user@example.com",
    "name": "User Name"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| valid | boolean | `true` if the session is valid and user has an active subscription |
| plan | string \| null | Current plan: `"free"`, `"pro"`, `"enterprise"` |
| status | string \| null | Subscription status: `"active"`, `"past_due"`, `"canceled"`, etc. |
| cancel_at_period_end | boolean \| null | `true` if subscription will cancel at end of billing period |
| expires_at | string \| null | ISO 8601 datetime when current billing period ends |
| user | object \| null | Basic user info |
| user.email | string | User's email |
| user.name | string \| null | Display name |

#### Response — 200 OK (free / unsubscribed user)

```json
{
  "valid": false,
  "plan": "free",
  "status": "inactive",
  "cancel_at_period_end": false,
  "expires_at": null,
  "user": {
    "email": "user@example.com",
    "name": "User Name"
  }
}
```

> Note: `valid: false` means the user is authenticated but does NOT have an active subscription. The client treats this as unsubscribed.

#### Error Responses

| Status | Meaning | Client Behavior |
|--------|---------|-----------------|
| 401 | Session token expired or invalid | Prompt user to re-login |
| 403 | User exists but has no subscription | Show subscribe URL. Response body may include `subscribe_url`: `{"valid":false,"error":"no_subscription","subscribe_url":"/cloud","plan":"free"}` |
| 5xx | Server error | Fall back to cached subscription status |

---

## 3. Model Catalog API

### `GET /api/v1/models`

Returns the list of cloud models available through tknet.ai. Used to populate the TokenHub provider list in NovaMLX.

#### Request

```
GET /api/v1/models
Authorization: Bearer <session-token>
```

| Header | Required | Description |
|--------|----------|-------------|
| Authorization | ✅ | `Bearer <session-token>` from login |

#### Response — 200 OK

```json
{
  "models": [
    {
      "id": "deepseek-v4-pro",
      "name": "DeepSeek V4 Pro",
      "show_in_tokenhub": true,
      "context_window": 1000000,
      "capabilities": {
        "input": ["text"],
        "output": ["text"]
      }
    },
    {
      "id": "glm-5.1",
      "name": "GLM 5.1",
      "show_in_tokenhub": true,
      "context_window": 1000000,
      "capabilities": {
        "input": ["text", "image"],
        "output": ["text"]
      }
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| models | array | List of available models |
| models[].id | string | Model identifier used in API calls. Must be unique. |
| models[].name | string | Human-readable display name |
| models[].show_in_tokenhub | boolean | **Critical**: If `true`, this model should be shown as a managed provider in NovaMLX TokenHub UI. If `false` or absent, the model exists in the catalog but should NOT be auto-provisioned as a TokenHub provider. |
| models[].context_window | integer | Maximum context window in tokens (optional, defaults to 128000) |
| models[].capabilities | object | Optional. Modality support. |
| models[].capabilities.input | string[] | Input types: `"text"`, `"image"` |
| models[].capabilities.output | string[] | Output types: `"text"` |

> **`show_in_tokenhub` is the key filtering field**. NovaMLX will only auto-create managed providers for models where `show_in_tokenhub == true`. This lets tknet.ai control which models appear in the TokenHub UI without removing them from the catalog.

#### Response — 401 Unauthorized

```json
{
  "error": "unauthorized",
  "message": "Invalid or expired session"
}
```

---

## 4. Inference Proxy API (existing, for reference)

NovaMLX's TokenHub already proxies inference requests to cloud providers. The model catalog entries' `id` field maps directly to the `model` field in inference API calls. No changes needed here — just documenting the flow:

```
Client → NovaMLX (localhost:6590) → tknet.ai inference endpoint → upstream model provider
```

The inference endpoint URL for tknet.ai managed models should be configurable. Suggested default:

```
POST https://tknet.ai/api/v1/chat/completions
POST https://tknet.ai/api/v1/responses
Authorization: Bearer <session-token>
```

These are standard OpenAI-compatible endpoints. tknet.ai should proxy them to the respective upstream providers.

---

## 5. Summary of API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/v1/auth/login` | POST | None | Authenticate user, get session token |
| `/api/v1/auth/check` | POST | Session in body | Validate session, get subscription status |
| `/api/v1/models` | GET | Bearer token | List available models with `show_in_tokenhub` filter |
| `/api/v1/chat/completions` | POST | Bearer token | Inference proxy (OpenAI-compatible) |
| `/api/v1/responses` | POST | Bearer token | Inference proxy (Responses API format) |

---

## 6. Client-Side Caching Behavior

NovaMLX caches auth state locally for offline / fast-path access:

- **Auth cache**: `~/.nova/auth_cache.json` — TTL 5 minutes. Contains `valid`, `plan`, `status`, `cancelAtPeriodEnd`, `expiresAt`, `userEmail`.
- **Session token**: `~/.nova/session` — File permission 600. Read on every `validate()` call.
- **Provider list**: Refreshed on app launch when `validate()` succeeds and `GET /models` returns.

If the auth cache is expired and network is unavailable, NovaMLX treats the user as **unsubscribed** (free tier: max 3 providers). This is a conservative fallback.

---

## 7. Migration Path

Current default auth URL: `https://novamlx.ai`
New default auth URL: `https://tknet.ai`

The auth URL is resolved in this priority order:
1. Environment variable `NOVA_AUTH_URL`
2. Config file `~/.nova/config.json` → `auth.authURL`
3. Hardcoded default (will be changed from `novamlx.ai` to `tknet.ai`)

Once tknet.ai implements the above APIs, we will change the hardcoded default and release an update. Existing users can also override via config file or env var for testing before the release.

---

## 8. Testing Checklist

For tknet.ai developers, verify these scenarios:

- [ ] `POST /api/v1/auth/login` with valid credentials → 200 + session token
- [ ] `POST /api/v1/auth/login` with wrong password → 401
- [ ] `POST /api/v1/auth/login` with non-existent email → 401
- [ ] `POST /api/v1/auth/check` with valid session → 200 + `valid: true` + plan info
- [ ] `POST /api/v1/auth/check` with expired session → 401
- [ ] `POST /api/v1/auth/check` for free-tier user → 200 with `valid: false` (or 403 with subscribe_url)
- [ ] `GET /api/v1/models` with valid Bearer token → 200 + model list
- [ ] `GET /api/v1/models` with models where `show_in_tokenhub: true` and `show_in_tokenhub: false` mixed
- [ ] `GET /api/v1/models` without auth → 401
- [ ] `POST /api/v1/chat/completions` with `model: "deepseek-v4-pro"` → streaming inference response
