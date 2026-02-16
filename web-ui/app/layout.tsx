import './globals.css'

export const metadata = {
  title: 'SphinxSkynet Blockchain',
  description: 'Production blockchain with mining and cross-chain bridge',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  )
}
