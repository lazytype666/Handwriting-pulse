import Constants, { AppOwnership } from 'expo-constants';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';

export type OcrSource = 'camera' | 'library';

export interface OcrCaptureResult {
  uri: string;
  mlkitText?: string;
  mode: 'mlkit_only' | 'vlm_only';
}

// Expo Go 不包含 rn-mlkit-ocr 的原生模块。现代 manifest 下 appOwnership 可能为 null，故同时校验 expoGoConfig。
export const IS_EXPO_GO =
  Constants.appOwnership === AppOwnership.Expo || Boolean(Constants.expoGoConfig);

const MAX_EDGE = 1280;
const PAGE = '请使用 EAS 开发构建或独立 APK 运行离线 OCR，或在 .env.local 配置商用 VLM API。';

interface Picked {
  uri: string;
  width: number;
  height: number;
}

/** 是否已配置商用/云端或局域网多模态视觉识题模型（VLM）；若只配了统一商用 LLM API 且为多模态模型，也可直接共用 */
export function hasConfiguredVlmEndpoint(): boolean {
  const vlmModel = process.env.EXPO_PUBLIC_VLM_MODEL?.trim();
  const sharedLlmModel = process.env.EXPO_PUBLIC_LLM_MODEL?.trim();
  const base =
    process.env.EXPO_PUBLIC_VLM_BASE_URL?.trim() ||
    process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim() ||
    process.env.EXPO_PUBLIC_EDGE_LLM_URL?.trim();
  const key =
    process.env.EXPO_PUBLIC_VLM_API_KEY?.trim() ||
    process.env.EXPO_PUBLIC_LLM_API_KEY?.trim();
  if (!base) return false;
  if (vlmModel) return true;
  if (key && sharedLlmModel && /vl|vision|gemini|gpt-4|gpt-5|claude|qvq|multimodal/i.test(sharedLlmModel)) {
    return true;
  }
  return false;
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
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.85,
      exif: false,
      // Android 使用内置裁剪界面；不传 aspect 时保持自由比例，便于框选题目区域。
      allowsEditing: true,
    });
    if (result.canceled) return null;
    const asset = result.assets[0];
    return asset ? { uri: asset.uri, width: asset.width, height: asset.height } : null;
  }
  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) throw permissionError(permission, '照片');
  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    quality: 0.85,
    exif: false,
    allowsEditing: true,
  });
  if (result.canceled) return null;
  const asset = result.assets[0];
  return asset ? { uri: asset.uri, width: asset.width, height: asset.height } : null;
}

// 限制长边并适当压缩，在保留数学符号清晰度的同时显著加快向商用/本地 VLM 上传并降低超时率。
async function shrink({ uri, width, height }: Picked): Promise<string> {
  const longest = Math.max(width, height);
  try {
    const context = ImageManipulator.manipulate(uri);
    if (longest > MAX_EDGE) {
      const scale = MAX_EDGE / longest;
      context.resize({ width: Math.round(width * scale), height: Math.round(height * scale) });
    }
    const image = await context.renderAsync();
    const saved = await image.saveAsync({ compress: 0.78, format: SaveFormat.JPEG });
    return saved.uri;
  } catch {
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

/** 拍照或相册选图、自由裁剪并缩放到安全尺寸，返回图片 URI（用户取消返回 null） */
export async function pickCroppedImageUri(source: OcrSource): Promise<string | null> {
  const picked = await pick(source);
  if (!picked) return null;
  return shrink(picked);
}

/** 使用设备端内置中文 ML Kit 执行纯本地文本识别（仅在未配置商用/本地 VLM 的 Dev Client/APK 中作为离线兜底） */
export async function recognizeRawTextWithMlKit(uri: string): Promise<string> {
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

/**
 * 统一视觉采题入口（商用 VLM 优先直通，彻底消除一维 ML Kit 压扁乱码干扰）：
 * - 已配置商用 VLM / 共用多模态 API / 局域网 VLM：直接裁剪压图（1280px）走 `vlm_only`，省去本地 ML Kit 延迟与错误 hint 干扰；
 * - 未配置 VLM 时：在 Dev Client / APK 下自动启用端侧 `rn-mlkit-ocr`；Expo Go 则提示配置 API Key。
 */
export async function captureProblemForIngest(
  source: OcrSource,
  existingCroppedUri?: string,
): Promise<OcrCaptureResult | null> {
  const vlmEnabled = hasConfiguredVlmEndpoint();
  if (IS_EXPO_GO && !vlmEnabled) {
    throw new Error('Expo Go 不包含设备端识别模块。' + PAGE);
  }

  const uri = existingCroppedUri ?? (await pickCroppedImageUri(source));
  if (!uri) return null;

  if (vlmEnabled || IS_EXPO_GO) {
    return { uri, mode: 'vlm_only' };
  }

  const mlkitText = await recognizeRawTextWithMlKit(uri);
  return { uri, mlkitText, mode: 'mlkit_only' };
}

// 兼容旧调用签名：拍照或选图后在本机用内置中文 ML Kit 模型识别。返回 null 表示用户取消。
export async function recognizeQuestion(source: OcrSource): Promise<string | null> {
  if (IS_EXPO_GO) throw new Error('Expo Go 不包含设备端识别模块。' + PAGE);
  const uri = await pickCroppedImageUri(source);
  if (!uri) return null;
  return recognizeRawTextWithMlKit(uri);
}
