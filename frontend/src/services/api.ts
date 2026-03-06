import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 2100000, // 35分钟超时（与后端 LLM_TIMEOUT=1800s 对齐，留余量）
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 通过 SSE 流式生成旅行计划（实时接收后端 Agent 进度）
 */
export async function generateTripPlanStream(
  formData: TripFormData,
  onProgress: (event: { step: string; percent: number; message: string; streamText?: string }) => void,
): Promise<TripPlanResponse> {
  const url = `${API_BASE_URL}/api/trip/plan/stream`

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData),
  })

  if (!response.ok) {
    const errText = await response.text().catch(() => '')
    throw new Error(errText || `HTTP ${response.status}`)
  }

  if (!response.body) {
    throw new Error('浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: TripPlanResponse | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const data = JSON.parse(line.slice(6))
        onProgress({ step: data.step, percent: data.percent, message: data.message, streamText: data.streamText })

        if (data.step === 'done' && data.data) {
          result = { success: true, message: '旅行计划生成成功', data: data.data }
        } else if (data.step === 'error') {
          throw new Error(data.error || data.message || '生成失败')
        }
      } catch (e: any) {
        if (e instanceof SyntaxError) continue // 跳过不完整的 JSON
        throw e
      }
    }
  }

  if (!result) {
    throw new Error('未收到完整的旅行计划数据')
  }

  return result
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient

/**
 * 批量获取景点图片
 */
export async function fetchAttractionPhotos(
  names: string[],
  city: string = ''
): Promise<Record<string, string>> {
  try {
    const response = await apiClient.post<{ success: boolean; data: Record<string, string> }>(
      '/api/trip/photos',
      { names, city }
    )
    return response.data.data || {}
  } catch (error: any) {
    console.error('获取景点图片失败:', error)
    return {}
  }
}

