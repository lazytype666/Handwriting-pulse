import * as FileSystem from 'expo-file-system/legacy';
import type { MotorFingerprint } from './cognitiveState';
import type { PolicySnapshot } from './personalizedPolicy';

export interface PersistedUserProfile {
  version: number;
  updatedAt: number;
  fingerprint: MotorFingerprint;
  policy: PolicySnapshot;
}

const PROFILE_FILENAME = 'inkpulse-user-profile.json';

function getProfileUri(): string | null {
  const dir = FileSystem.documentDirectory;
  return dir ? `${dir}${PROFILE_FILENAME}` : null;
}

/**
 * 本地持久化用户的内环运动指纹与外环 LinUCB 个性化权重。
 * 【开源库调用】使用 expo-file-system 执行应用文档目录内安全读写。
 */
export async function loadPersistedProfile(): Promise<PersistedUserProfile | null> {
  const uri = getProfileUri();
  if (!uri) return null;
  try {
    const info = await FileSystem.getInfoAsync(uri);
    if (!info.exists) return null;
    const raw = await FileSystem.readAsStringAsync(uri);
    const parsed = JSON.parse(raw) as PersistedUserProfile;
    if (!parsed || typeof parsed !== 'object' || !parsed.fingerprint || !parsed.policy) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function savePersistedProfile(profile: PersistedUserProfile): Promise<void> {
  const uri = getProfileUri();
  if (!uri) return;
  const pending = `${uri}.part`;
  try {
    await FileSystem.writeAsStringAsync(pending, JSON.stringify(profile));
    await FileSystem.moveAsync({ from: pending, to: uri });
  } catch {
    await FileSystem.deleteAsync(pending, { idempotent: true }).catch(() => {});
  }
}

export async function clearPersistedProfile(): Promise<void> {
  const uri = getProfileUri();
  if (!uri) return;
  await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
}
