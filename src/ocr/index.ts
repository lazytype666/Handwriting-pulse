import Constants, { AppOwnership } from 'expo-constants';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';

export type OcrSource = 'camera' | 'library';

export interface OcrCaptureResult {
  uri: string;
  mlkitText?: string;
  mode: 'mlkit_only' | 'vlm_only' | 'mlkit_with_vlm';
}

// Expo Go 不包含 rn-mlkit-ocr 的原生模块。现代 manifest 下 appOwnership 可能为 null，故同时校验 expoGoConfig。
export const IS_EXPO_GO =
  Constants.appOwnership === AppOwnership.Expo || Boolean(Constants.expoGoConfig);

const MAX_EDGE = 1280;
const PAGE = '请使用 EAS 开发构建或独立 APK 运行识题功能。';

interface Picked {
  uri: string;
  width: number;
  height: number;
}

/** 是否已配置端侧/局域网或云端多模态视觉识题模型（VLM） */
export function hasConfiguredVlmEndpoint(): boolean {
  const vlmModel = process.env.EXPO_PUBLIC_VLM_MODEL?.trim();
  const base =
    process.env.EXPO_PUBLIC_VLM_BASE_URL?.trim() ||
    process.env.EXPO_PUBLIC_EDGE_LLM_URL?.trim() ||
    process.env.EXPO_PUBLIC_LLM_BASE_URL?.trim();
  return Boolean(vlmModel && base);
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

// 限制长边并适当压缩，在保留数学符号清晰度的同时显著加快热点向本地 VLM 上传并降低超时率。
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

/** 拍照或相册选图、自由裁剪并缩放到安全尺寸，返回图片 URI（用户取消返回 null） */
export async function pickCroppedImageUri(source: OcrSource): Promise<string | null> {
  const picked = await pick(source);
  if (!picked) return null;
  return shrink(picked);
}

/** 使用设备端内置中文 ML Kit 执行纯本地文本识别 */
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
 * 统一视觉采题入口：
 * - 若未配 VLM 且在 Expo Go：提示需使用 EAS 开发构建或 APK；
 * - 若在开发构建/APK：本地先跑 ML Kit OCR，若同时配置 VLM（如本地 qwen2.5vl），则返回双通道上下文供 LaTeX 精修；
 * - 若在 Expo Go 但已配置 VLM：支持直接裁剪选图并交由本地/云端 VLM 输出标准 LaTeX 题干。
 */
export async function captureProblemForIngest(source: OcrSource): Promise<OcrCaptureResult | null> {
  const vlmEnabled = hasConfiguredVlmEndpoint();
  if (IS_EXPO_GO && !vlmEnabled) {
    throw new Error('Expo Go 不包含设备端识别模块。' + PAGE);
  }

  const uri = await pickCroppedImageUri(source);
  if (!uri) return null;

  if (IS_EXPO_GO) {
    return { uri, mode: 'vlm_only' };
  }

  try {
    const mlkitText = await recognizeRawTextWithMlKit(uri);
    return {
      uri,
      mlkitText,
      mode: vlmEnabled ? 'mlkit_with_vlm' : 'mlkit_only',
    };
  } catch (err) {
    if (vlmEnabled) {
      return { uri, mode: 'vlm_only' };
    }
    throw err;
  }
}

// 兼容旧调用签名：拍照或选图后在本机用内置中文 ML Kit 模型识别。返回 null 表示用户取消。
export async function recognizeQuestion(source: OcrSource): Promise<string | null> {
  if (IS_EXPO_GO) throw new Error('Expo Go 不包含设备端识别模块。' + PAGE);
  const uri = await pickCroppedImageUri(source);
  if (!uri) return null;
  return recognizeRawTextWithMlKit(uri);
}
