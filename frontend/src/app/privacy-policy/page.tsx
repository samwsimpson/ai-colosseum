// frontend/src/app/privacy-policy/page.tsx
'use client';

import React from 'react';

export default function PrivacyPolicyPage() {
  return (
    <div className="flex-grow flex flex-col items-center justify-center p-8 md:p-12 text-white">
      <div className="w-full max-w-4xl bg-gray-900 rounded-2xl shadow-xl p-8 md:p-12">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-6 text-center">
          Privacy Policy
        </h1>
        <div className="prose prose-invert max-w-none text-gray-400 space-y-4">
          <p>
            This Privacy Policy describes how your personal information is collected, used, and shared when you visit or make a purchase from The AI Colosseum (the “Site”).
          </p>
          <h3>Personal Information We Collect</h3>
          <p>
            When you visit the Site, we automatically collect certain information about your device, including information about your web browser, IP address, time zone, and some of the cookies that are installed on your device.
          </p>
          <h3>How We Use Your Personal Information</h3>
          <p>
            We use the Personal Information we collect to help us screen for potential risk and fraud (in particular, your IP address), and more generally to improve and optimize our Site (for example, by generating analytics about how our customers browse and interact with the Site, and to assess the success of our marketing and advertising campaigns).
          </p>
          <h3>Your Rights</h3>
          <p>
            If you are a European resident, you have the right to access personal information we hold about you and to ask that your personal information be corrected, updated, or deleted. If you would like to exercise this right, please contact us through the contact information below.
          </p>
          <h3>Contact Us</h3>
          <p>
            For more information about our privacy practices, if you have questions, or if you would like to make a complaint, please contact us by e-mail at kitt@kumokodo.ai.
          </p>
        </div>
      </div>
    </div>
  );
}
