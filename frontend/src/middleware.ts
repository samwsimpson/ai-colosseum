import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const PROTECTED = ['/chat']; // add more protected roots if needed

export function middleware(req: NextRequest) {
  const url = req.nextUrl;
  const path = url.pathname;

  // Only guard protected paths
  if (!PROTECTED.some(p => path.startsWith(p))) {
    return NextResponse.next();
  }

  // We rely on the HttpOnly refresh cookie your backend sets.
  // If it's missing, the user is not (or no longer) logged in.
  const hasRefresh = req.cookies.get('refresh_token');
  if (!hasRefresh) {
    const signIn = url.clone();
    signIn.pathname = '/sign-in';
    // Preserve destination
    signIn.searchParams.set('next', path + (url.search || ''));
    return NextResponse.redirect(signIn);
  }

  // Let it through
  return NextResponse.next();
}

// Only run middleware on /chat/**
export const config = {
  matcher: ['/chat/:path*'],
};
