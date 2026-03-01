"use client"

import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Upload, X, Image } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ImageUploadCardProps {
  onImageUpload: (file: File) => void
  onClear?: () => void
  selectedImage?: File | null
  imagePreview?: string | null
  disabled?: boolean
  className?: string
}

export default function ImageUploadCard({
  onImageUpload,
  onClear,
  selectedImage,
  imagePreview,
  disabled = false,
  className
}: ImageUploadCardProps) {
  const [dragActive, setDragActive] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (disabled) return

    const files = e.dataTransfer.files
    if (files && files[0]) {
      handleFile(files[0])
    }
  }

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file')
      return
    }

    onImageUpload(file)
    
    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files[0]) {
      handleFile(files[0])
    }
  }

  const clearImage = () => {
    setPreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    if (onClear) {
      onClear()
    }
  }

  return (
    <Card className={cn("h-full shadow-none border-0", className)}>
      <CardContent className="p-0 h-full">
        <div
          className={cn(
            "relative border-2 border-dashed rounded-xl h-full min-h-[280px] flex items-center justify-center transition-all duration-300",
            dragActive ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20 scale-[1.02]" : "border-gray-300 dark:border-gray-700",
            disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-950/10",
            preview || imagePreview ? "p-4" : "p-8"
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <Input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            className="hidden"
            disabled={disabled}
          />
          
          {preview || imagePreview ? (
            <div className="space-y-3 w-full flex flex-col items-center pointer-events-none">
              <div className="relative w-full flex justify-center">
                <img
                  src={imagePreview || preview || ''}
                  alt="Preview"
                  className="max-w-full max-h-60 object-contain rounded-lg shadow-md"
                />
                <Button
                  variant="destructive"
                  size="sm"
                  className="absolute top-2 right-2 rounded-full shadow-lg h-8 w-8 p-0 z-10 cursor-pointer pointer-events-auto hover:scale-110 transition-transform"
                  onClick={(e) => {
                    e.stopPropagation()
                    clearImage()
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 text-center font-medium truncate px-4 max-w-full pointer-events-none">
                {selectedImage?.name || 'Image uploaded'}
              </p>
            </div>
          ) : (
            <div className="text-center space-y-4">
              <div className="bg-blue-100 dark:bg-blue-900/30 w-16 h-16 rounded-xl flex items-center justify-center mx-auto">
                <Image className="h-8 w-8 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="space-y-2">
                <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                  Drop image here or click to browse
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  JPG, PNG, GIF up to 10MB
                </p>
              </div>
              <Button variant="outline" size="sm" disabled={disabled} className="rounded-lg text-sm h-9 px-4">
                <Upload className="mr-2 h-4 w-4" />
                Browse Files
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}