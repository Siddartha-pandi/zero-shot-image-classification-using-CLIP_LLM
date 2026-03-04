import axios, { AxiosInstance } from 'axios'
import type { ClassificationResult, AddClassResponse } from '@/types'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

class APIClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: BACKEND_URL,
      timeout: 60000, // 60 second timeout for inference
    })
  }

  /**
   * Classify an image with hybrid routing (auto domain detection + MedCLIP for medical)
   */
  async classifyImage(
    imageFile: File,
    userText?: string
  ): Promise<ClassificationResult> {
    const formData = new FormData()
    formData.append('file', imageFile)
    if (userText) {
      formData.append('custom_labels', userText)
    }
    formData.append('top_k', '10')

    const response = await this.client.post<ClassificationResult>(
      '/api/classify-hybrid',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )

    return response.data
  }

  /**
   * Open-ended classification (extracts objects from caption)
   */
  async classifyOpenEnded(
    imageFile: File,
    userText?: string
  ): Promise<ClassificationResult> {
    const formData = new FormData()
    formData.append('file', imageFile)
    if (userText) {
      formData.append('user_text', userText)
    }

    const response = await this.client.post<ClassificationResult>(
      '/api/classify',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )

    return response.data
  }

  /**
   * Add a new class for classification
   */
  async addClass(
    label: string,
    domain?: string,
    images?: File[]
  ): Promise<AddClassResponse> {
    const formData = new FormData()
    formData.append('label', label)
    if (domain) {
      formData.append('domain', domain)
    }
    if (images && images.length > 0) {
      images.forEach((file) => {
        formData.append('files', file)
      })
    }

    const response = await this.client.post<AddClassResponse>(
      '/api/add-class',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )

    return response.data
  }

  /**
   * Get list of all registered classes
   */
  async getClasses(): Promise<string[]> {
    const response = await this.client.get<{ classes: string[] }>(
      '/api/classes'
    )
    return response.data.classes
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get<{ status: string }>('/health')
    return response.data
  }

  /**
   * Evaluate dataset
   */
  async evaluateDataset(
    imageFiles: File[],
    labels: string[]
  ): Promise<any> {
    const formData = new FormData()
    imageFiles.forEach((file) => {
      formData.append('files', file)
    })
    labels.forEach((label) => {
      formData.append('labels', label)
    })

    const response = await this.client.post<any>(
      '/api/evaluate',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )

    return response.data
  }
}

export const apiClient = new APIClient()
