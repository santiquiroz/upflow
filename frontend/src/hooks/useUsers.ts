import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createUser,
  getUserJobs,
  listUsers,
  updateUser,
  type CreateUserParams,
  type UpdateUserParams,
} from "../services/users";

export function useUsers() {
  return useQuery({ queryKey: ["users"], queryFn: listUsers });
}

export function useCreateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: CreateUserParams) => createUser(params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["users"] }),
  });
}

export function useUpdateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, params }: { userId: string; params: UpdateUserParams }) => updateUser(userId, params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["users"] }),
  });
}

export function useUserJobs(userId: string | null) {
  return useQuery({
    queryKey: ["userJobs", userId],
    queryFn: () => getUserJobs(userId as string),
    enabled: userId !== null,
  });
}
