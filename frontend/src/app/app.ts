import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { ReactiveFormsModule, FormBuilder, FormGroup } from '@angular/forms';
import { inject } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

export interface EntityOption {
  key: string;
  label: string;
  description: string;
}

export type RedactionState = 'idle' | 'uploading' | 'processing' | 'done' | 'error';

const ALL_ENTITIES: EntityOption[] = [
  { key: 'PERSON',                label: 'Person Names',        description: 'Full or partial names of individuals' },
  { key: 'EMAIL_ADDRESS',         label: 'Email Addresses',     description: 'Personal or business email addresses' },
  { key: 'PHONE_NUMBER',          label: 'Phone Numbers',       description: 'US and international phone numbers' },
  { key: 'US_SSN',                label: 'Social Security #',   description: 'US Social Security Numbers' },
  { key: 'CREDIT_CARD',           label: 'Credit Card #',       description: 'Credit and debit card numbers' },
  { key: 'US_ADDRESS',            label: 'Street Addresses',    description: 'US mailing and street addresses' },
  { key: 'ZIP_CODE',              label: 'ZIP Codes',           description: 'US postal ZIP codes' },
  { key: 'AGE',                   label: 'Ages',                description: 'Numeric age references' },
  { key: 'LOCATION',              label: 'Locations',           description: 'Cities, states, countries, and landmarks' },
  { key: 'STRICT_DATE',           label: 'Dates',               description: 'Calendar dates in various formats' },
  { key: 'MEDICAL_RECORD_NUMBER', label: 'Medical Record #',    description: 'Hospital MRN identifiers' },
];

@Component({
  selector: 'app-root',
  imports: [ReactiveFormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class App {
  private fb = inject(FormBuilder);
  private sanitizer = inject(DomSanitizer);

  readonly entities = ALL_ENTITIES;
  readonly selectedFile = signal<File | null>(null);
  readonly state = signal<RedactionState>('idle');
  readonly errorMessage = signal<string>('');
  readonly downloadUrl = signal<string>('');
  readonly downloadName = signal<string>('');
  readonly isDragOver = signal(false);

  readonly safePreviewUrl = computed<SafeResourceUrl | null>(() => {
    const url = this.downloadUrl();
    return url ? this.sanitizer.bypassSecurityTrustResourceUrl(url) : null;
  });

  readonly form: FormGroup = this.fb.group(
    Object.fromEntries(ALL_ENTITIES.map(e => [e.key, true]))
  );

  readonly selectedCount = computed(() => {
    const vals = this.form.value as Record<string, boolean>;
    return Object.values(vals).filter(Boolean).length;
  });

  readonly canSubmit = computed(() =>
    this.selectedFile() !== null &&
    this.selectedCount() > 0 &&
    this.state() !== 'uploading' &&
    this.state() !== 'processing'
  );

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.setFile(file);
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver.set(true);
  }

  onDragLeave(): void {
    this.isDragOver.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver.set(false);
    const file = event.dataTransfer?.files[0] ?? null;
    this.setFile(file);
  }

  private setFile(file: File | null): void {
    if (file && file.type !== 'application/pdf') {
      this.state.set('error');
      this.errorMessage.set('Only PDF files are supported.');
      return;
    }
    this.selectedFile.set(file);
    this.state.set('idle');
    this.errorMessage.set('');
    this.downloadUrl.set('');
  }

  selectAll(): void {
    const patch = Object.fromEntries(ALL_ENTITIES.map(e => [e.key, true]));
    this.form.patchValue(patch);
  }

  selectNone(): void {
    const patch = Object.fromEntries(ALL_ENTITIES.map(e => [e.key, false]));
    this.form.patchValue(patch);
  }

  async submit(): Promise<void> {
    const file = this.selectedFile();
    if (!file || !this.canSubmit()) return;

    const entities = ALL_ENTITIES
      .filter(e => this.form.value[e.key])
      .map(e => e.key);

    this.state.set('uploading');
    this.errorMessage.set('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('entities', JSON.stringify(entities));

    try {
      this.state.set('processing');
      const response = await fetch('/api/redact', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Server error: ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const baseName = file.name.replace(/\.pdf$/i, '');
      this.downloadUrl.set(url);
      this.downloadName.set(`${baseName}_REDACTED.pdf`);
      this.state.set('done');
    } catch (err: unknown) {
      this.state.set('error');
      this.errorMessage.set(err instanceof Error ? err.message : 'An unexpected error occurred.');
    }
  }

  reset(): void {
    if (this.downloadUrl()) {
      URL.revokeObjectURL(this.downloadUrl());
    }
    this.selectedFile.set(null);
    this.state.set('idle');
    this.errorMessage.set('');
    this.downloadUrl.set('');
    this.downloadName.set('');
    this.selectAll();
  }
}
