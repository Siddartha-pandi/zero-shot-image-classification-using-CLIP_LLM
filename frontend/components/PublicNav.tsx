"use client"

import { ThemeToggle } from "@/components/theme-toggle"

export function PublicNav() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-black dark:border-white bg-white dark:bg-black">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Left side - Logo */}
        <div className="flex items-center gap-2">
          <div className="hidden sm:block">
            <h1 className="font-bold text-lg text-black dark:text-white">CLIP-LLM</h1>
            <p className="text-xs text-gray-600 dark:text-gray-400">Zero-Shot Image Classification</p>
          </div>
        </div>

        {/* Right side - Theme toggle */}
        <div className="flex items-center gap-3">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
