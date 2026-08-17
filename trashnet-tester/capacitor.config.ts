import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  // El appId por defecto del starter es io.ionic.starter, que choca con
  // cualquier otra app de Ionic sin configurar que haya en el teléfono.
  appId: 'cl.equipodinamita.trashnettester',
  appName: 'TrashNet Tester',
  webDir: 'www',
};

export default config;
