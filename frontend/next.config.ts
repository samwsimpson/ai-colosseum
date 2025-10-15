// next.config.ts
import type { NextConfig } from 'next';

const CSP = [
  "default-src 'self'",

  // Allow API + WebSocket + Stripe + Google auth calls
  "connect-src 'self' https://api.aicolosseum.app wss://api.aicolosseum.app https://m.stripe.network https://accounts.google.com https://oauth2.googleapis.com https://www.googleapis.com",

  // Scripts: allow inline (for now), Stripe, and Google Identity Services
  "script-src 'self' 'unsafe-inline' https://js.stripe.com https://m.stripe.network https://accounts.google.com https://accounts.google.com/gsi/client",

  // Styles: allow inline + your cdnjs stylesheet
  "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com",

  // Images: local + data/blob + Google user avatars
  "img-src 'self' data: blob: https://*.googleusercontent.com",

  // Fonts: allow local data: and cdnjs Font Awesome webfonts
  "font-src 'self' data: https://cdnjs.cloudflare.com",

  // Frames: allow Stripe + Google auth popups/iframes
  "frame-src 'self' https://js.stripe.com https://accounts.google.com",

  "base-uri 'self'",
  "frame-ancestors 'self'",
].join('; ');



const securityHeaders = [
  { key: 'Content-Security-Policy', value: CSP },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
  { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },

  // Keep this from your original file:
  { key: 'Cross-Origin-Opener-Policy', value: 'same-origin-allow-popups' },
  // (Do NOT set Cross-Origin-Embedder-Policy unless you need SAB/WebGPU; it can break Stripe/OAuth if misconfigured.)
] satisfies { key: string; value: string }[];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: '/:path*',
        headers: securityHeaders,
      },
    ];
  },
};

export default nextConfig;
