"use client"

import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Upload, X, Image } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ImageUploadCardProps {
  onImageUpload: (file: File) => void
  selectedImage?: File | null
  disabled?: boolean
  className?: string
}

export default function ImageUploadCard({
  onImageUpload,
  selectedImage,
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
    // Call onImageUpload with null or undefined to clear
  }

  return (
    <Card className={cn(" h-full shadow-none border-0", className)}>
      <CardContent className="p-0 h-full">
        <div
          className={cn(
            "relative border-2 border-dashed rounded-xl h-full flex items-center justify-center transition-all duration-300",
            dragActive ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20 scale-[1.02]" : "border-gray-300 dark:border-gray-700",
            disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-950/10",
            preview ? "p-3" : "p-6"
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
          
          {preview ? (
            <div className="space-y-2 w-full">
              <div className="relative">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full max-h-52 object-contain rounded-lg"
                />
                <Button
                  variant="destructive"
                  size="sm"
                  className="absolute top-1 right-1 rounded-full shadow-lg h-7 w-7 p-0"
                  onClick={(e) => {
                    e.stopPropagation()
                    clearImage()
                  }}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 text-center truncate px-2">
                {selectedImage?.name || 'Image uploaded'}
              </p>
            </div>
          ) : (
            <div className="text-center space-y-3">
              <div className="bg-blue-100 dark:bg-blue-900/30 w-12 h-12 rounded-xl flex items-center justify-center mx-auto">
                <Image className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="space-y-1">
                <p className="text-xs font-semibold text-gray-800 dark:text-gray-200">
                  Drop image here or click to browse
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  JPG, PNG, GIF up to 10MB
                </p>
              </div>
              <Button variant="outline" size="sm" disabled={disabled} className="rounded-lg text-xs h-8">
                <Upload className="mr-2 h-3 w-3" />
                Browse Files
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
