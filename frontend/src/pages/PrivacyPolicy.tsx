import { Navbar } from '../components/layout/Navbar';

const PrivacyPolicy = (): JSX.Element => {
  return (
  <>
    <Navbar />

    <section className="min-h-screen bg-[#020202] text-white px-4 sm:px-6 lg:px-8 py-24">
      <div className="max-w-5xl mx-auto">

        {/* Heading */}
        <div className="mb-16 text-center">
          <h1 className="text-4xl font-bold mb-4">
            Privacy <span className="text-blue-600">Policy</span>
          </h1>
          <p className="text-gray-500 text-sm max-w-2xl mx-auto">
            Your privacy matters. This page explains how DriveDetect handles data and transparency.
          </p>
        </div>

        <div className="space-y-8">

          {/* 1 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">1. Introduction</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              DriveDetect is an open-source computer vision research initiative.
              We are committed to ensuring transparency in how data is handled.
              This Privacy Policy outlines what information is collected, how it
              is used, and how users remain in control.
            </p>
          </div>

          {/* 2 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">2. Data Collection</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              DriveDetect does not collect, store, or transmit personal user data
              to external servers. Any image, video, or sensor input processed
              during live demonstrations remains local to the user's device.
              No biometric, identity, or behavioral data is retained.
            </p>
          </div>

          {/* 3 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">3. Local Processing</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              All AI inference and driver monitoring analysis occur locally
              within the user's runtime environment. The system is designed
              for research and experimentation purposes and does not perform
              remote cloud-based tracking.
            </p>
          </div>

          {/* 4 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">4. Third-Party Services</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              DriveDetect may contain links to third-party platforms such as GitHub.
              These platforms operate independently and have their own privacy
              policies. We encourage users to review those policies separately.
            </p>
          </div>

          {/* 5 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">5. Security Practices</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              While DriveDetect does not store personal data, we follow best
              practices in open-source development to maintain code integrity
              and transparency. Users are encouraged to run the software in
              secure environments and review source code before deployment.
            </p>
          </div>

          {/* 6 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">6. Policy Updates</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              This Privacy Policy may be updated as the project evolves.
              Updates will reflect improvements in architecture, features,
              or integrations. Continued use of DriveDetect implies acceptance
              of the most recent version of this policy.
            </p>
          </div>

        </div>
      </div>
    </section>
  </>
);
};

export default PrivacyPolicy;