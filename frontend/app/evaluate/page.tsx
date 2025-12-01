'use client'

import { useState } from 'react'

export default function EvaluatePage() {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)

  const handleEvaluate = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      const data = await response.json()
      setResults(data)
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-6">Model Evaluation</h1>
      <button 
        onClick={handleEvaluate} 
        disabled={isLoading}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {isLoading ? 'Loading...' : 'Evaluate Model'}
      </button>
      {results && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Results:</h2>
          <pre className="bg-gray-100 p-4 rounded overflow-auto">
            {JSON.stringify(results, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
