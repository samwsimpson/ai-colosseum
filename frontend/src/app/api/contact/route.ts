// frontend/src/app/api/contact/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { Resend } from 'resend'; // CORRECTED: Named import for the Resend class

const resend = new Resend(process.env.RESEND_API_KEY);
const RECIPIENT_EMAIL = process.env.RECIPIENT_EMAIL;
const FROM_EMAIL = process.env.FROM_EMAIL;
const RECAPTCHA_SECRET_KEY = process.env.RECAPTCHA_SECRET_KEY;

export async function POST(req: NextRequest) {
  try {
    const { name, email, message, recaptchaToken } = await req.json();

    if (!name || !email || !message || !recaptchaToken) {
      return NextResponse.json({ message: 'Missing required fields' }, { status: 400 });
    }

    // Verify the reCAPTCHA token with Google
    const recaptchaRes = await fetch('https://www.google.com/recaptcha/api/siteverify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `secret=${RECAPTCHA_SECRET_KEY}&response=${recaptchaToken}`,
    });

    const recaptchaData = await recaptchaRes.json();

    if (!recaptchaData.success) {
      console.error('reCAPTCHA verification failed:', recaptchaData['error-codes']);
      return NextResponse.json({ message: 'reCAPTCHA verification failed' }, { status: 400 });
    }

    if (!RECIPIENT_EMAIL || !FROM_EMAIL) {
      console.error('Environment variables for email not set.');
      return NextResponse.json({ message: 'Email service not configured.' }, { status: 500 });
    }

    const { data, error } = await resend.emails.send({
      from: `Contact Form <${FROM_EMAIL}>`,
      to: [RECIPIENT_EMAIL],
      subject: `New message from ${name}`,
      html: `
        <p><strong>Name:</strong> ${name}</p>
        <p><strong>Email:</strong> ${email}</p>
        <p><strong>Message:</strong> ${message}</p>
      `,
    });

    if (error) {
      console.error('Error from Resend:', error);
      return NextResponse.json({ message: 'Failed to send message.' }, { status: 500 });
    }

    return NextResponse.json({ message: 'Message sent successfully!', data }, { status: 200 });

  } catch (error) {
    console.error('Error processing contact form:', error);
    return NextResponse.json({ message: 'Failed to send message.' }, { status: 500 });
  }
}
