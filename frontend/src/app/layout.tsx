// frontend/src/app/layout.tsx
import { Inter } from 'next/font/google';
import './globals.css';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { Header } from '../components/Header';
import Footer from '../components/Footer';
import React from 'react';
import { UserProvider } from '../context/UserContext';

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
  //const clientId = '418819543347-8aes37r4191r49df7dnsue14p049orls.apps.googleusercontent.com';
  
  if (!clientId) {
    return (
      <html lang="en">
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
            {/* The header is now fixed and will stay at the top */}
            <Header />
            {/* The `main` element now fills the space between header and footer */}
            <main className="flex-1 pt-20">
              {children}
            </main>
            <Footer />
          </UserProvider>
        </GoogleOAuthProvider>
      </body>
    </html>
  );
}
