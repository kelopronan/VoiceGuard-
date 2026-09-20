/** @type {import('next').NextConfig} */
let rawBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || 'http://localhost:8000';

// Ensure it starts with http:// or https://
if (rawBackendUrl && !rawBackendUrl.startsWith('http://') && !rawBackendUrl.startsWith('https://')) {
  rawBackendUrl = `https://${rawBackendUrl}`;
}

const backendUrl = (rawBackendUrl.replace(/\/+$/, '')) || 'http://localhost:8000';

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

