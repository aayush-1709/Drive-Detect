import { Footer } from "../components/layout/Footer";

const OpenSource = (): JSX.Element => {
  return (
    <div className="min-h-screen bg-[#020202] text-white">

      <section className="px-4 sm:px-6 lg:px-8 py-24">
        <div className="max-w-4xl mx-auto">

          {/* Page Title */}
          <h1 className="text-4xl font-bold mb-8">
            Open Source <span className="text-blue-600">Project</span>
          </h1>

          {/* Project Description */}
          <div className="space-y-6 text-gray-400 text-sm leading-relaxed">

            <p>
              DriveDetect is an open-source project focused on building an
              AI-powered traffic sign recognition system using deep learning
              and computer vision technologies.
            </p>

            <p>
              The project encourages students, developers, and researchers
              to explore machine learning, improve user interfaces, and
              contribute to real-world intelligent transportation systems.
            </p>

            {/* Contribution Section */}
            <div className="pt-8">
              <h2 className="text-xl font-semibold text-white mb-4">
                How to Contribute
              </h2>

              <ul className="list-disc pl-6 space-y-2">
                <li>Fork the repository on GitHub</li>
                <li>Clone your fork locally</li>
                <li>Create a new feature branch for your changes</li>
                <li>Implement improvements or fix issues</li>
                <li>Commit and push your changes</li>
                <li>Open a Pull Request for review</li>
              </ul>
            </div>

            {/* Beginner Contribution Note */}
            <p className="pt-6">
              New contributors are encouraged to explore beginner-friendly
              issues and enhancements in the repository. Even small improvements
              such as UI fixes or documentation updates are valuable.
            </p>

            {/* GitHub Link */}
            <a
              href="https://github.com/aayush-1709/Drive-Detect"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block text-blue-500 hover:text-blue-400 transition pt-4 font-medium"
            >
              → View Repository on GitHub
            </a>

            {/* Community Note */}
            <p className="text-gray-500 pt-8">
              DriveDetect supports open-source collaboration and welcomes
              contributors from student developer programs, research groups,
              and the broader developer community.
            </p>

          </div>
        </div>
      </section>

      <Footer />

    </div>
  );
};

export default OpenSource;