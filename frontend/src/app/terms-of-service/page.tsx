// frontend/src/app/terms-of-service/page.tsx
'use client';

import React from 'react';

export default function TermsOfServicePage() {
  return (
    <div className="flex-grow flex flex-col items-center justify-center p-8 md:p-12 text-white">
      <div className="w-full max-w-4xl bg-gray-900 rounded-2xl shadow-xl p-8 md:p-12">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-6 text-center">
          Terms of Service
        </h1>
        <div className="prose prose-invert max-w-none text-gray-400 space-y-4">
          <p>
            Welcome to The AI Colosseum, these terms and conditions outline the rules and regulations for the use of our website.
          </p>
          <h3>1. Acceptance of Terms</h3>
          <p>
            By accessing or using our Service, you agree to be bound by these Terms. If you disagree with any part of the terms, then you may not access the Service.
          </p>
          <h3>2. Use of Service</h3>
          <p>
            The Service and its original content, features, and functionality are and will remain the exclusive property of KumoKodo.ai and its licensors. The Service is protected by copyright, trademark, and other laws of both the United States and foreign countries.
          </p>
          <h3>3. Termination</h3>
          <p>
            We may terminate or suspend access to our Service immediately, without prior notice or liability, for any reason whatsoever, including without limitation if you breach the Terms.
          </p>
          <h3>4. Governing Law</h3>
          <p>
            These Terms shall be governed and construed in accordance with the laws of Texas, without regard to its conflict of law provisions.
          </p>
          <h3>5. Changes to Terms</h3>
          <p>
            We reserve the right, at our sole discretion, to modify or replace these Terms at any time. What constitutes a material change will be determined at our sole discretion.
          </p>
        </div>
      </div>
    </div>
  );
}
