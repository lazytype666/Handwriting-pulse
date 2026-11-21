import Constants, { AppOwnership } from 'expo-constants';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';

export type OcrSource = 'camera' | 'library';

// Expo Go 不包含 rn-mlkit-ocr 的原生模块。开发构建与独立 APK 的 appOwnership 为 null。
export const IS_EXPO_GO = Constants.appOwnership === AppOwnership.Expo;

const MAX_EDGE = 2400;
const PAGE = '请使用 EAS 开发构建或独立 APK 运行识题功能。';

interface Picked {
  uri: string;
  width: number;
  height: number;
}

function permissionError(permission: ImagePicker.PermissionResponse, what: string): Error {
  return new Error(
    permission.canAskAgain
      ? `需要${what}权限才能识别题目，请在系统弹窗中允许。`
      : `${what}权限已关闭。请在系统设置中允许笔迹脉搏使用${what}后重试。`,
  );
}

async function pick(source: OcrSource): Promise<Picked | null> {
  if (source === 'camera') {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) throw permissionError(permission, '相机');
    const result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.85, exif: false });
    if (result.canceled) return null;
    const asset = result.assets[0];
    return asset ? { uri: asset.uri, width: asset.width, height: asset.height } : null;
  }
  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) throw permissionError(permission, '照片');
  const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85, exif: false });
  if (result.canceled) return null;
  const asset = result.assets[0];
  return asset ? { uri: asset.uri, width: asset.width, height: asset.height } : null;
}

// 端侧 OCR 不需要保留原始分辨率。缩小长边可显著降低解码内存，并让识别更快。
async function shrink({ uri, width, height }: Picked): Promise<string> {
  const longest = Math.max(width, height);
  if (!longest || longest <= MAX_EDGE) return uri;
  const scale = MAX_EDGE / longest;
  try {
    const context = ImageManipulator.manipulate(uri);
    context.resize({ width: Math.round(width * scale), height: Math.round(height * scale) });
    const image = await context.renderAsync();
    const saved = await image.saveAsync({ compress: 0.85, format: SaveFormat.JPEG });
    return saved.uri;
  } catch {
    // 缩放失败时交给原生识别器处理原图，避免因为预处理让整次识别失败。
    return uri;
  }
}

function clean(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function isLinkingError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes('rn-mlkit-ocr') || message.includes('seem to be linked');
}

// 拍照或选图后在本机用内置中文 ML Kit 模型识别。返回 null 表示用户取消。
export async function recognizeQuestion(source: OcrSource): Promise<string | null> {
  if (IS_EXPO_GO) throw new Error('Expo Go 不包含设备端识别模块。' + PAGE);
  const picked = await pick(source);
  if (!picked) return null;
  const uri = await shrink(picked);

  // 延迟加载原生模块，避免 Expo Go 在启动时就触碰不存在的 NativeModule。
  const mlkit = require('rn-mlkit-ocr') as typeof import('rn-mlkit-ocr');
  let text: string;
  try {
    const result = await mlkit.recognizeText(uri, 'chinese');
    text = clean(result.text);
  } catch (error) {
    if (isLinkingError(error)) throw new Error('设备端识别模块未加载。' + PAGE);
    throw error;
  }
  if (!text) throw new Error('没有识别到文字。请让题目占满画面、保持光线充足后重试。');
  return text;
}
