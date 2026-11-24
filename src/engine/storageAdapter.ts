import * as FileSystem from 'expo-file-system/legacy';
import type { LearnerCognitiveTwin } from '../agent';
import type { MotorFingerprint } from './cognitiveState';
import type { PolicySnapshot } from './personalizedPolicy';
import { loadPersistedProfile, savePersistedProfile } from './profileStore';
import type { SolutionCacheEntry } from './solutionCache';
import type { EngineSessionPayload, IStorageAdapter } from './types';

const CACHE_FILENAME = 'inkpulse-solution-cache.json';
const LAST_SESSION_FILENAME = 'inkpulse-last-session-summary.json';
const LEARNER_TWIN_FILENAME = 'inkpulse-learner-twin.json';

function docUri(filename: string): string | null {
  const dir = FileSystem.documentDirectory;
  return dir ? `${dir}${filename}` : null;
}

async function atomicWriteJson(uri: string, payload: unknown): Promise<void> {
  const pending = `${uri}.part`;
  try {
    await FileSystem.writeAsStringAsync(pending, JSON.stringify(payload));
    await FileSystem.moveAsync({ from: pending, to: uri });
  } catch {
    await FileSystem.deleteAsync(pending, { idempotent: true }).catch(() => {});
  }
}

/**
 * 本地持久化存储适配器（实现 IStorageAdapter）。
 * 【开源库调用】基于 expo-file-system 实现画像、解题图谱缓存与会话摘要原子落盘；
 * 预留依赖注入接口，便于后续平滑扩展云同步或团队后端存储。
 */
export class LocalStorageAdapter implements IStorageAdapter {
  public async loadUserProfile(_userId: string): Promise<{
    fingerprint: MotorFingerprint;
    policy: PolicySnapshot;
  } | null> {
    const saved = await loadPersistedProfile();
    if (!saved) return null;
    return { fingerprint: saved.fingerprint, policy: saved.policy };
  }

  public async saveUserProfile(
    _userId: string,
    profile: { fingerprint: MotorFingerprint; policy: PolicySnapshot },
  ): Promise<void> {
    await savePersistedProfile({
      version: 1,
      updatedAt: Date.now(),
      fingerprint: profile.fingerprint,
      policy: profile.policy,
    });
  }

  public async saveSession(payload: EngineSessionPayload): Promise<void> {
    const uri = docUri(LAST_SESSION_FILENAME);
    if (!uri) return;
    await atomicWriteJson(uri, payload);
  }

  public async loadSolutionCache(): Promise<SolutionCacheEntry[]> {
    const uri = docUri(CACHE_FILENAME);
    if (!uri) return [];
    try {
      const info = await FileSystem.getInfoAsync(uri);
      if (!info.exists) return [];
      const raw = await FileSystem.readAsStringAsync(uri);
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as SolutionCacheEntry[]) : [];
    } catch {
      return [];
    }
  }

  public async saveSolutionCache(entries: SolutionCacheEntry[]): Promise<void> {
    const uri = docUri(CACHE_FILENAME);
    if (!uri || entries.length === 0) return;
    // 控制端侧图谱缓存数量在最近 40 道题内，避免存储膨胀
    const recent = [...entries]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, 40);
    await atomicWriteJson(uri, recent);
  }

  public async loadLearnerTwin(): Promise<LearnerCognitiveTwin | null> {
    const uri = docUri(LEARNER_TWIN_FILENAME);
    if (!uri) return null;
    try {
      const info = await FileSystem.getInfoAsync(uri);
      if (!info.exists) return null;
      const raw = await FileSystem.readAsStringAsync(uri);
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' && parsed.schemaVersion === 'twin-v1'
        ? (parsed as LearnerCognitiveTwin)
        : null;
    } catch {
      return null;
    }
  }

  public async saveLearnerTwin(twin: LearnerCognitiveTwin): Promise<void> {
    const uri = docUri(LEARNER_TWIN_FILENAME);
    if (!uri) return;
    await atomicWriteJson(uri, twin);
  }
}
