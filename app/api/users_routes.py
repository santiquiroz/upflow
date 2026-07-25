# app/api/users_routes.py
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.auth_deps import get_user_store, require
from app.schemas import (
    CreateUserRequest,
    CreateUserResponse,
    OwnedJobSummaryResponse,
    UpdateUserRequest,
    UpdateUserResponse,
    UserJobsResponse,
    UserSummaryResponse,
    UsersListResponse,
)
from app.services.auth.identity import AuthenticatedUser, authenticated_user_from_record
from app.services.auth.passwords import generate_salt, hash_password
from app.services.auth.permissions import Permission, Role
from app.services.auth.quotas import QuotaService
from app.services.auth.user_store import User, UserStore

router = APIRouter(prefix="/api/v1/users", tags=["users"])


def get_quota_service(request: Request) -> QuotaService:
    return request.app.state.quota_service


def _user_summary(user: User, quotas: QuotaService) -> UserSummaryResponse:
    status = quotas.status_for(authenticated_user_from_record(user))
    return UserSummaryResponse(
        id=user.id, username=user.username, role=user.role.value, disabled=user.disabled,
        must_change_password=user.must_change_password, quota_overrides=user.quota_overrides,
        created_at=user.created_at, used_jobs_today=status.used_jobs_today,
        used_gpu_seconds_today=status.used_gpu_seconds_today,
    )


@router.get("", response_model=UsersListResponse)
async def list_users(
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
    _current_user: AuthenticatedUser = Depends(require(Permission.users_manage)),
) -> UsersListResponse:
    return UsersListResponse(users=[_user_summary(u, quotas) for u in user_store.list()])


@router.post("", response_model=CreateUserResponse, status_code=201)
async def create_user(
    payload: CreateUserRequest,
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
    _current_user: AuthenticatedUser = Depends(require(Permission.users_manage)),
) -> CreateUserResponse:
    try:
        role = Role(payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown role: {payload.role!r}") from exc
    temporary_password = secrets.token_urlsafe(12)
    salt = generate_salt()
    password_hash = hash_password(temporary_password, salt)
    try:
        user = user_store.create(
            username=payload.username, password_hash=password_hash, salt=salt,
            role=role, must_change_password=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CreateUserResponse(user=_user_summary(user, quotas), temporary_password=temporary_password)


def _apply_user_update(user_store: UserStore, user_id: str, payload: UpdateUserRequest) -> User:
    user = user_store.get(user_id)
    if user is None:
        raise ValueError(f"Unknown user id: {user_id!r}")
    if payload.role is not None:
        user = user_store.set_role(user_id, Role(payload.role))
    if payload.disabled is not None:
        user = user_store.set_disabled(user_id, payload.disabled)
    if payload.quota_overrides is not None:
        user = user_store.set_quota_overrides(user_id, payload.quota_overrides)
    return user


@router.patch("/{user_id}", response_model=UpdateUserResponse)
async def update_user(
    user_id: str,
    payload: UpdateUserRequest,
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
    _current_user: AuthenticatedUser = Depends(require(Permission.users_manage)),
) -> UpdateUserResponse:
    try:
        user = _apply_user_update(user_store, user_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    temporary_password = None
    if payload.reset_password:
        temporary_password = secrets.token_urlsafe(12)
        salt = generate_salt()
        password_hash = hash_password(temporary_password, salt)
        user = user_store.set_password(user_id, password_hash, salt, must_change_password=True)
    return UpdateUserResponse(user=_user_summary(user, quotas), temporary_password=temporary_password)


@router.get("/{user_id}/jobs", response_model=UserJobsResponse)
async def get_user_jobs(
    user_id: str,
    request: Request,
    _current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_all)),
) -> UserJobsResponse:
    managers = (
        ("image", request.app.state.job_manager),
        ("video", request.app.state.video_job_manager),
        ("audio", request.app.state.audio_job_manager),
        ("generation", request.app.state.generation_job_manager),
    )
    jobs = [
        OwnedJobSummaryResponse(
            id=job.id, kind=kind, status=job.status,
            original_filename=getattr(job, "original_filename", None),
            created_at=job.created_at, finished_at=job.finished_at,
        )
        for kind, manager in managers
        for job in manager.jobs.values()
        if job.owner_id == user_id
    ]
    return UserJobsResponse(jobs=jobs)
