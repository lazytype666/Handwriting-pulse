import * as FileSystem from 'expo-file-system/legacy';
import type { Stroke } from '../capture';
import type { StallEvent } from '../stalls';

export interface ExportReceipt {
  uri: string;
  strokeCount: number;
  pointCount: number;
  stallCount: number;
  annotationCount?: number;
}

let sequence = 0;
let pickerOpen = false;

async function nextFileUri(date: Date): Promise<string> {
  const directory = FileSystem.documentDirectory;
  if (!directory) throw new Error('应用文档目录不可用');

  const stamp = date.toISOString().replace(/[:.]/g, '-');
  // 同毫秒的导出也有独立名称。重启后若遇到已有文件，继续寻找空位。
  for (;;) {
    sequence += 1;
    const uri = directory + 'inkpulse-' + stamp + '-' + sequence + '.json';
    if (!(await FileSystem.getInfoAsync(uri)).exists) return uri;
  }
}

export async function saveSnapshot(
  strokes: Stroke[],
  stallEvents: StallEvent[] = [],
  cognitiveSession?: {
    problemId?: string;
    problemText?: string;
    annotations?: unknown[];
    stateTrace?: unknown[];
    [key: string]: unknown;
  },
): Promise<ExportReceipt> {
  const now = new Date();
  const strokeCount = strokes.length;
  const pointCount = strokes.reduce((sum, stroke) => sum + stroke.points.length, 0);
  const annotations = Array.isArray(cognitiveSession?.annotations)
    ? cognitiveSession.annotations
    : undefined;
  const stateTrace = Array.isArray(cognitiveSession?.stateTrace)
    ? cognitiveSession.stateTrace
    : undefined;

  const data = JSON.stringify({
    meta: {
      exportedAt: now.toISOString(),
      strokeCount,
      pointCount,
      ...(cognitiveSession ? { version: '2.0-annotated' } : {}),
      ...(cognitiveSession?.problemId ? { problemId: cognitiveSession.problemId } : {}),
    },
    strokes,
    stallEvents,
    ...(annotations ? { annotations } : {}),
    ...(stateTrace ? { stateTrace } : {}),
    ...(cognitiveSession ? { cognitiveSession } : {}),
  });
  const uri = await nextFileUri(now);
  const pending = uri + '.part';
  // 开源 expo-file-system 负责本地 I/O。先完整写入临时文件，再在同目录改名。
  try {
    await FileSystem.writeAsStringAsync(pending, data);
    await FileSystem.moveAsync({ from: pending, to: uri });
  } catch (error) {
    await FileSystem.deleteAsync(pending, { idempotent: true }).catch(() => {});
    throw error;
  }
  return {
    uri,
    strokeCount,
    pointCount,
    stallCount: stallEvents.length,
    ...(annotations ? { annotationCount: annotations.length } : {}),
  };
}

export async function copyToFolder(receipt: ExportReceipt): Promise<string | null> {
  if (pickerOpen) throw new Error('请先完成当前的文件夹选择');
  pickerOpen = true;
  const saf = FileSystem.StorageAccessFramework;
  let destination: string | undefined;
  try {
    const permission = await saf.requestDirectoryPermissionsAsync();
    if (!permission.granted) return null;
    const data = await FileSystem.readAsStringAsync(receipt.uri);
    // Android 通过 MIME 类型确定扩展名，SAF 要求传入不带扩展名的名称。
    const basename = receipt.uri.slice(receipt.uri.lastIndexOf('/') + 1, -5);
    destination = await saf.createFileAsync(permission.directoryUri, basename, 'application/json');
    await FileSystem.writeAsStringAsync(destination, data);
    return destination;
  } catch (error) {
    // 仅清理本次新建但写入失败的副本。保留应用文档目录中的完整原件以便重试。
    if (destination) {
      await FileSystem.deleteAsync(destination, { idempotent: true }).catch(() => {});
    }
    throw error;
  } finally {
    pickerOpen = false;
  }
}
