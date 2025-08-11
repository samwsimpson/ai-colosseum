'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { useGoogleLogin } from '@react-oauth/google';
import { useUser } from '@/context/UserContext';

export default function SignInPage() {
  const { userToken, handleLogin } = useUser();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [isLoading, setIsLoading] = useState(false);
  const processedRef = useRef(false); // prevents double POSTs after redirect

  const backendUrl =
    process.env.NEXT_PUBLIC_WS_URL?.replace(/\/+$/, '') || 'http://localhost:8000';

  // Single handler used by both redirect callback & button fallback
  const handleGoogleSuccess = useCallback(
    async (tokenResponse: { code: string; redirect_uri: string }) => {
      if (processedRef.current) return;
      processedRef.current = true;

      setIsLoading(true);
      try {
        const res = await fetch(`${backendUrl}/api/google-auth`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code: tokenResponse.code,
            redirect_uri: tokenResponse.redirect_uri,
          }),
        });

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(
            `Google authentication failed on backend. Details: ${JSON.stringify(errorData)}`
          );
        }

        const tokenData = await res.json(); // expects { access_token, user_name? }
        handleLogin(tokenData);
        router.push('/chat');
      } catch (err) {
        console.error('Sign-in failed:', err);
        setIsLoading(false);
        processedRef.current = false; // allow retry
      }
    },
    [backendUrl, handleLogin, router]
  );

  // Handle Google redirect callback (?code=...)
  useEffect(() => {
    if (userToken) return; // already signed in
    const code = searchParams.get('code');
    if (!code) return;

    const redirect_uri = `${window.location.origin}${pathname}`; // must match Google OAuth config
    handleGoogleSuccess({ code, redirect_uri });
  }, [searchParams, pathname, userToken, handleGoogleSuccess]);

  // If already signed-in, bounce to chat
  useEffect(() => {
    if (userToken) router.replace('/chat');
  }, [userToken, router]);

  // Google button (auth-code + redirect flow)
  const login = useGoogleLogin({
    flow: 'auth-code',
    ux_mode: 'redirect',
    // This must exactly match one of the Authorized redirect URIs in your Google credentials
    redirect_uri: typeof window !== 'undefined' ? `${window.location.origin}/sign-in` : undefined,
    // scopes: ['openid', 'profile', 'email'], // optional; defaults usually fine
    onError: (err) => {
      console.error('Google init error:', err);
    },
  });

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-neutral-950 text-white">
      <main className="w-full max-w-md p-6 rounded-2xl bg-neutral-900 shadow-lg">
        <h1 className="text-2xl font-semibold mb-2">Welcome back</h1>
        <p className="text-sm text-neutral-400 mb-8">
          Sign in with Google to continue to Colosseum.
        </p>

        <button
          onClick={() => {
            if (!isLoading) login(); // triggers Google redirect
          }}
          disabled={isLoading}
          className="w-full inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 bg-white text-black font-medium hover:bg-neutral-200 disabled:opacity-60"
          aria-busy={isLoading}
        >
          <svg className="w-5 h-5" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
            <path
              fill="#FFC107"
              d="M43.611 20.082H42V20H24v8h11.305C33.876 31.657 29.35 35 24 35c-7.18 0-13-5.82-13-13s5.82-13 13-13c3.31 0 6.31 1.23 8.59 3.24l5.66-5.66C34.84 3.02 29.65 1 24 1 11.85 1 2 10.85 2 23s9.85 22 22 22c11 0 21-8 21-22 0-1.39-.2-2.81-.49-4.14z"
            />
            <path
              fill="#4285F4"
              d="M6.305 14.691l6.571 4.817C14.33 16.377 18.8 12 24 12c3.31 0 6.31 1.23 8.59 3.24l5.66-5.66C34.84 3.02 29.65 1 24 1 15.27 1 7.64 5.64 3.34 12.36l2.965 2.331z"
            />
            <path
              fill="#FBBC05"
              d="M24 45c5.29 0 10.13-1.94 13.86-5.26l-6.4-5.24C29.9 35.46 27.15 36.5 24 36.5c-5.32 0-9.84-3.61-11.45-8.47l-6.63 5.12C9.2 40.63 16.09 45 24 45z"
            />
            <path
              fill="#EA4335"
              d="M43.611 20.082H42V20H24v8h11.305c-1.21 3.157-4.22 5.5-7.305 5.5-5.32 0-9.84-3.61-11.45-8.47l-6.63 5.12C11.47 37.75 17.38 41 24 41c11 0 21-8 21-22 0-1.39-.2-2.81-.49-4.14z"
            />
          </svg>
          {isLoading ? 'Signing in…' : 'Sign in with Google'}
        </button>

        <p className="mt-6 text-xs text-neutral-500">
          By continuing, you agree to our Terms and Privacy Policy.
        </p>
      </main>
    </div>
  );
}
