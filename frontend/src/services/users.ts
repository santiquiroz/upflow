import { apiGet, apiPatchJson, apiPostJson } from "../lib/api";
import type {
  CreateUserResponse,
  UpdateUserResponse,
  UserJobsResponse,
  UsersListResponse,
} from "../lib/apiTypes";

export interface CreateUserParams {
  username: string;
  role: string;
}

export interface UpdateUserParams {
  role?: string;
  disabled?: boolean;
  quotaOverrides?: Record<string, number>;
  resetPassword?: boolean;
}

export function listUsers(): Promise<UsersListResponse> {
  return apiGet<UsersListResponse>("/users");
}

export function createUser(params: CreateUserParams): Promise<CreateUserResponse> {
  return apiPostJson<CreateUserResponse>("/users", params);
}

export function updateUser(userId: string, params: UpdateUserParams): Promise<UpdateUserResponse> {
  return apiPatchJson<UpdateUserResponse>(`/users/${userId}`, params);
}

export function getUserJobs(userId: string): Promise<UserJobsResponse> {
  return apiGet<UserJobsResponse>(`/users/${userId}/jobs`);
}
