// next.config.ts
import type { NextConfig } from 'next';

const CSP = [
  "default-src 'self'",
  // REST + WebSocket + Stripe telemetry
  "connect-src 'self' https://api.aicolosseum.app wss://api.aicolosseum.app https://m.stripe.network",
  // IMPORTANT: allow inline scripts so Next.js and your page can run
  "script-src 'self' 'unsafe-inline' https://js.stripe.com https://m.stripe.network",
  // Allow inline styles + your cdnjs stylesheet
  "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  // Stripe iframes
  "frame-src 'self' https://js.stripe.com",
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
