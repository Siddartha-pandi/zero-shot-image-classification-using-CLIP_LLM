"use client"

import { SessionProvider } from "next-auth/react"
import { LandingPageTopNav } from "@/components/LandingPageTopNav"

export function LandingPageNav() {
  return (
    <SessionProvider>
      <LandingPageTopNav />
    </SessionProvider>
  )
}
