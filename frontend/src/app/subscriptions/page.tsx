'use client';

import { useState, useEffect } from 'react';
import React from 'react';
import { useUser } from '../../context/UserContext';
import { useRouter } from 'next/navigation';
import { loadStripe } from '@stripe/stripe-js';

// Load Stripe.js with your publishable key
// Replace 'YOUR_STRIPE_PUBLISHABLE_KEY' with your actual Stripe Publishable Key
const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY || 'YOUR_STRIPE_PUBLISHABLE_KEY');

// Placeholder for your Stripe Price IDs
// You must replace these with the actual Price IDs from your Stripe Dashboard
const STRIPE_PRICE_IDS = {
  starter: process.env.NEXT_PUBLIC_STRIPE_PRICE_STARTER || 'starter_price_id_placeholder',
  pro:     process.env.NEXT_PUBLIC_STRIPE_PRICE_PRO     || 'pro_price_id_placeholder',
};

interface PlanCardProps {
  title: string;
  price: string;
  features: string[];
  buttonText: string;
  isEnterprise?: boolean;
  priceId?: string;
  isCurrentPlan: boolean;
  conversationsUsed?: number | null;
  monthlyLimit?: number | null;
  resetsOn?: string | null;   // NEW
}

const PlanCard: React.FC<PlanCardProps> = ({
  title,
  price,
  features,
  buttonText,
  isEnterprise = false,
  priceId = '',
  isCurrentPlan,
  conversationsUsed,
  monthlyLimit,
  resetsOn,       // NEW
}) => {
  const { userToken } = useUser();
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubscribe = async () => {
    if (!userToken) {
      router.push('/sign-in');
      return;
    }

    if (isEnterprise) {
      router.push('/contact-us');
      return;
    }

    setIsSubmitting(true);
    try {
      const rawApi = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
      const backendUrl = rawApi.replace(/^wss:\/\//, 'https://').replace(/^ws:\/\//, 'http://');

      const response = await fetch(`${backendUrl}/api/create-checkout-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${userToken}`,
        },
        body: JSON.stringify({ price_id: priceId }),
      });

      if (!response.ok) {
        throw new Error('Failed to create checkout session.');
      }

      const session = await response.json();
      const stripe = await stripePromise;
      if (stripe) {
        await stripe.redirectToCheckout({ sessionId: session.id });
      }
    } catch (error) {
      console.error('Error creating checkout session:', error);
      setIsSubmitting(false);
    }
  };

  return (
    <div className={`relative flex flex-col p-6 rounded-3xl shadow-xl transition-all duration-300 ease-in-out transform hover:scale-105 ${isEnterprise ? 'bg-blue-600 text-white' : 'bg-gray-800 text-white'}`}>
      {isCurrentPlan && (
        <span className="absolute top-0 right-0 mt-4 mr-4 bg-green-500 text-gray-900 text-xs font-semibold px-3 py-1 rounded-full uppercase">
          Current Plan
        </span>
      )}
      <h3 className={`text-2xl font-bold ${isEnterprise ? 'text-white' : 'text-blue-500'}`}>{title}</h3>
      <p className={`mt-4 text-4xl font-extrabold ${isEnterprise ? 'text-white' : 'text-gray-100'}`}>{price}</p>
      
      {isCurrentPlan && (
        monthlyLimit === null ? (
          <p className="mt-2 text-sm text-gray-400">
            Unlimited conversations{resetsOn ? ` • Renews on ${resetsOn}` : ''}
          </p>
        ) : (
          <p className="mt-2 text-sm text-gray-400">
            You&apos;ve used {conversationsUsed ?? 0} of {monthlyLimit} conversations{resetsOn ? ` • Resets on ${resetsOn}` : ''}.
          </p>
        )
      )}

      <ul className="mt-6 flex-1 space-y-3 text-gray-300">
        {features.map((feature, index) => (
          <li key={index} className="flex items-center space-x-3">
            <svg className="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <span>{feature}</span>
          </li>
        ))}
      </ul>
      <button
        onClick={handleSubscribe}
        disabled={isSubmitting || isCurrentPlan}
        className={`mt-8 w-full py-3 px-6 rounded-xl text-lg font-semibold shadow-lg transition-colors duration-200 ${
          isEnterprise
            ? 'bg-white text-blue-600 hover:bg-gray-200'
            : isCurrentPlan
            ? 'bg-gray-500 text-white cursor-not-allowed'
            : 'bg-blue-600 text-white hover:bg-blue-700'
        } disabled:opacity-50`}
      >
        {isSubmitting ? 'Processing...' : buttonText}
      </button>
    </div>
  );
};

export default function SubscriptionPage() {
  const { userToken } = useUser();
  const router = useRouter();
  const [currentPlanName, setCurrentPlanName] = useState<string | null>(null);
  const [conversationsUsed, setConversationsUsed] = useState<number | null>(null);
  const [monthlyLimit, setMonthlyLimit] = useState<number | null>(null);
  const [resetsOn, setResetsOn] = useState<string | null>(null);

  useEffect(() => {
    if (!userToken) {
      router.push('/sign-in');
      return;
    }

    const fetchCurrentPlan = async () => {
      try {
        const rawApi = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
        const backendUrl = rawApi.replace(/^wss:\/\//, 'https://').replace(/^ws:\/\//, 'http://');
        
        const userResponse = await fetch(`${backendUrl}/api/users/me`, {
          headers: {
            'Authorization': `Bearer ${userToken}`,
          },
        });
        const usageResponse = await fetch(`${backendUrl}/api/users/me/usage`, {
          headers: {
            'Authorization': `Bearer ${userToken}`,
          },
        });

        if (userResponse.ok && usageResponse.ok) {
          const userData = await userResponse.json();
          const usageData = await usageResponse.json();
          
          setCurrentPlanName(userData.user_plan_name || 'Free');
          setConversationsUsed(usageData.monthly_usage);
          setMonthlyLimit(usageData.monthly_limit);
          try {
            if (userData?.billing_period_end) {
              const d = new Date(userData.billing_period_end);
              // Format for your users; tweak locale/options as you like
              setResetsOn(d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }));
            } else {
              setResetsOn(null);
            }
          } catch {
            setResetsOn(null);
          }


        } else {
          console.error("Failed to fetch user data or usage data.");
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };
    fetchCurrentPlan();
  }, [userToken, router]);

  if (!userToken) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans antialiased py-12 md:py-24">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-extrabold text-white">Choose Your Plan</h1>
          <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">
            Select the perfect plan for your needs. Upgrade or contact us for custom solutions.
          </p>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-8 max-w-6xl mx-auto mb-16">
            <PlanCard
                title="Free"
                price="$0/month"
                features={[
                    "Access to core features",
                    "AI-to-AI critique",
                    "Limited to 5 conversations/month",
                    "Community support"
                ]}
                buttonText={currentPlanName === 'Free' ? "Current Plan" : "Get Started"}
                isCurrentPlan={currentPlanName === 'Free'}
                conversationsUsed={conversationsUsed}
                monthlyLimit={monthlyLimit}
                resetsOn={resetsOn}
            />
            <PlanCard
                title="Starter"
                price="$29/month"
                features={[
                    "Access to all core features",
                    "Multi-AI orchestration",
                    "Up to 25 conversations/month",
                    "Community support"
                ]}
                buttonText={currentPlanName === 'Starter' ? "Current Plan" : "Get Started"}
                isCurrentPlan={currentPlanName === 'Starter'}
                priceId={STRIPE_PRICE_IDS.starter}
                conversationsUsed={conversationsUsed}
                monthlyLimit={monthlyLimit}
                resetsOn={resetsOn}
            />
            <PlanCard
                title="Pro"
                price="$149/month"
                features={[
                    "Everything in Starter",
                    "Up to 200 conversations/month",
                    "Priority API access",
                    "Advanced API integrations",
                    "Team collaboration tools"
                ]}
                buttonText={currentPlanName === 'Pro' ? "Current Plan" : "Upgrade"}
                isCurrentPlan={currentPlanName === 'Pro'}
                priceId={STRIPE_PRICE_IDS.pro}
                conversationsUsed={conversationsUsed}
                monthlyLimit={monthlyLimit}
                resetsOn={resetsOn}
            />
             <PlanCard
                title="Enterprise"
                price="Custom"
                features={[
                    "Everything in Pro",
                    "Dedicated infrastructure",
                    "Custom AI model integrations",
                    "Volume discounts",
                    "SAML SSO & Audit logs",
                    "Dedicated technical support"
                ]}
                buttonText="Contact Us"
                isEnterprise
                isCurrentPlan={currentPlanName === 'Enterprise'}
                conversationsUsed={conversationsUsed}
                monthlyLimit={monthlyLimit}
                resetsOn={resetsOn}
            />
        </div>

        <section className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-blue-400">Why The AI Colosseum?</h2>
            <p className="mt-4 text-lg text-gray-300 max-w-3xl mx-auto">
                Unlike a single AI model, The Colosseum orchestrates a team of the world&apos;s most advanced AIs, including <strong>ChatGPT, Claude, Gemini, and Mistral</strong>. They collaborate, critique each other&apos;s work, and combine their unique strengths to deliver more comprehensive, accurate, and creative solutions. This isn&apos;t just about getting an answer—it&apos;s about getting the best answer from a diverse, expert team.
            </p>
        </section>

      </div>
    </div>
  );
}
