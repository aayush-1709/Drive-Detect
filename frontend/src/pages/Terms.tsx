import { Footer } from "../components/layout/Footer";
import { Navbar } from '../components/layout/Navbar';

const Terms = (): JSX.Element => {
  return (
  <>
    <Navbar />

    <section className="min-h-screen bg-[#020202] text-white px-4 sm:px-6 lg:px-8 py-24">
      <div className="max-w-5xl mx-auto">

        {/* Heading */}
        <div className="mb-16 text-center">
          <h1 className="text-4xl font-bold mb-4">
            Terms <span className="text-blue-600">of Use</span>
          </h1>
          <p className="text-gray-500 text-sm max-w-2xl mx-auto">
            Please read these terms carefully before using DriveDetect.
          </p>
        </div>

        <div className="space-y-8">

          {/* 1 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">1. Educational Purpose</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              DriveDetect is an open-source project created for educational,
              research, and experimental purposes. It is not certified for
              commercial or safety-critical deployment.
            </p>
          </div>

          {/* 2 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">2. User Responsibility</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              Users are solely responsible for how they implement, deploy,
              or modify DriveDetect. Any integration into vehicles,
              hardware systems, or real-world environments must comply
              with local laws and safety regulations.
            </p>
          </div>

          {/* 3 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">3. Limitation of Liability</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              The authors, contributors, and maintainers are not liable for
              damages, data loss, hardware malfunction, legal consequences,
              or misuse resulting from this project. Use the software at
              your own risk.
            </p>
          </div>

          {/* 4 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">4. Open-Source License</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              DriveDetect is distributed under its respective open-source
              license. Users may use, modify, and distribute the project
              in compliance with the license terms.
            </p>
          </div>

          {/* 5 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">5. Modifications & Contributions</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              Contributions are welcome. By submitting code or suggestions,
              you agree that your contributions may be included in the project
              under its existing license structure.
            </p>
          </div>

          {/* 6 */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-8 hover:border-blue-600/40 transition">
            <h2 className="text-xl font-semibold mb-4">6. Acceptance of Terms</h2>
            <p className="text-gray-400 text-sm leading-relaxed">
              Continued use of DriveDetect indicates that you understand and
              agree to these Terms of Use.
            </p>
          </div>

        </div>
      </div>
    </section>

    <Footer />
  </>
);
};

export default Terms;