// frontend/src/app/layout.tsx
import { Inter } from 'next/font/google';
import './globals.css';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { UserProvider } from '../context/UserContext';
import LayoutWrapper from '../components/LayoutWrapper';
import React from 'react';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'The Colosseum',
  description: 'AI Chat Application',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

  if (!clientId) {
    return (
      <html lang="en">
        <head>
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" />
        </head>        
        <body className="flex flex-col min-h-screen bg-gray-950 text-white">
          <main className="flex-grow flex flex-col items-center justify-center p-24">
            <h1 className="text-4xl font-bold mb-4">The Colosseum</h1>
            <p className="text-center text-red-400">
              Error: NEXT_PUBLIC_GOOGLE_CLIENT_ID is not set. Please check your .env.local file.
            </p>
          </main>
        </body>
      </html>
    );
  }

  return (
    <html lang="en">
      <body className={`flex flex-col min-h-screen bg-gray-950 ${inter.className}`}>
        <GoogleOAuthProvider clientId={clientId}>
          <UserProvider>
            <LayoutWrapper>
              {children}
            </LayoutWrapper>
          </UserProvider>
        </GoogleOAuthProvider>
      </body>
    </html>
  );
}
