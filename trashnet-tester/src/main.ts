import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { defineCustomElements } from '@ionic/pwa-elements/loader';

import { AppModule } from './app/app.module';

platformBrowserDynamic().bootstrapModule(AppModule)
  .catch(err => console.log(err));

// En dispositivo, @capacitor/camera usa la cámara nativa. En el navegador de
// escritorio necesita estos custom elements para mostrar el diálogo de cámara;
// sin esto, `CameraSource.Camera` falla en `ionic serve`.
defineCustomElements(window);
