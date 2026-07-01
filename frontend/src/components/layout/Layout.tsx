import type { ReactNode } from 'react';
import { Sidebar } from './Sidebar';

/**
 * Layout Component
 *
 * Root layout providing the application shell with sidebar navigation
 * and main content area. All pages render within this layout.
 */

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}
