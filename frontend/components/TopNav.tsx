"use client"

import { ThemeToggle } from "@/components/theme-toggle"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { Brain, User, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { signOut } from "next-auth/react"
import { useSession } from "next-auth/react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export function TopNav() {
  const { data: session } = useSession()

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Left side - Sidebar trigger and Logo */}
        <div className="flex items-center gap-4">
          <SidebarTrigger />
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary" />
            <div className="hidden sm:block">
              <h1 className="font-bold text-lg">CLIP-LLM</h1>
            </div>
          </div>
        </div>

        {/* Right side - Theme toggle and Profile */}
        <div className="flex items-center gap-3">
          <ThemeToggle />
          
          {session?.user && (
            <TooltipProvider>
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="h-9 w-9 rounded-full">
                      {session.user.image ? (
                        <img 
                          src={session.user.image} 
                          alt={session.user.name || "User"} 
                          className="h-9 w-9 rounded-full"
                        />
                      ) : (
                        <User className="h-5 w-5" />
                      )}
                      <span className="sr-only">User profile</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{session.user.name || session.user.email}</p>
                  </TooltipContent>
                </Tooltip>
                
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      onClick={() => signOut({ callbackUrl: "/" })}
                      className="h-9 w-9"
                    >
                      <LogOut className="h-5 w-5" />
                      <span className="sr-only">Sign out</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Sign out</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </TooltipProvider>
          )}
        </div>
      </div>
    </header>
  )
}
