# Desain Autentikasi dan Riwayat Analisis Berbasis Supabase

**Tanggal:** 12 Juli 2026  
**Status:** Disetujui untuk implementasi

## Ringkasan

Aplikasi Job Scam Detection akan mewajibkan autentikasi sebelum pengguna dapat mengakses model deteksi. Supabase Auth menangani sign up, verifikasi email, login, pemulihan password, refresh sesi, dan logout. Supabase Postgres menyimpan riwayat analisis per pengguna. Row Level Security (RLS) menjadi batas otorisasi utama agar setiap pengguna hanya dapat menambahkan dan membaca riwayatnya sendiri.

Pilihan Supabase menggantikan opsi awal SQLite karena lebih siap untuk deployment publik dan menyediakan alur email autentikasi tanpa implementasi SMTP di dalam aplikasi. SMTP bawaan Supabase hanya dipakai untuk demo skripsi dengan alamat anggota tim proyek. Custom SMTP tetap menjadi prasyarat operasional sebelum aplikasi dibuka untuk publik.

## Tujuan

- Menyediakan sign up dengan nama lengkap, email, password, dan konfirmasi password.
- Memverifikasi email sebelum akun dipakai untuk login.
- Menyediakan login, refresh sesi, logout, dan reset password melalui email.
- Menutup seluruh fitur deteksi bagi pengguna yang belum terautentikasi.
- Menyimpan teks lowongan lengkap dan hasil analisis untuk setiap pengguna.
- Menampilkan daftar riwayat dan detailnya secara read-only.
- Memastikan data satu pengguna tidak dapat dibaca atau ditulis atas nama pengguna lain.
- Menjaga struktur MVC yang sudah digunakan repository.

## Di Luar Cakupan

- Login sosial, anonymous login, MFA, role admin, dan organisasi multi-tenant.
- Opsi remember me lintas browser atau perangkat.
- Edit, hapus per item, dan hapus semua riwayat melalui UI.
- Penghapusan akun melalui UI.
- Penyimpanan gambar asli yang diunggah.
- Custom SMTP untuk tahap demo skripsi.

## Arsitektur

### Komponen baru

- `src/models/supabase_client.py`: membaca konfigurasi dan membuat client Supabase yang terisolasi per sesi Streamlit.
- `src/models/auth_service.py`: membungkus operasi Supabase Auth, pemulihan sesi, refresh token, dan pemetaan error.
- `src/models/history_repository.py`: menambahkan dan membaca data `analysis_history` menggunakan JWT pengguna aktif.
- `src/views/auth_view.py`: merender form login, sign up, lupa password, dan password baru.
- `src/views/history_view.py`: merender daftar berhalaman dan detail riwayat.

### Komponen yang berubah

- `src/controllers/app_controller.py`: memeriksa autentikasi sebelum memuat classifier, mengatur navigasi Analyze/History, menyimpan hasil analisis, dan menangani logout.
- `src/views/main_view.py`: tetap bertanggung jawab atas UI analisis; hanya penyesuaian kecil dilakukan bila diperlukan untuk mengembalikan sumber input dan data hasil yang akan disimpan.
- `requirements.txt`: menambahkan paket `supabase` dengan versi yang dikunci.
- `.gitignore`: memastikan file secrets lokal tidak pernah dilacak.
- Dokumentasi setup: menjelaskan konfigurasi Supabase, secrets, URL, template email, dan batas SMTP demo.

### Batas tanggung jawab

`AppController` hanya mengorkestrasi alur. `AuthService` tidak merender UI dan `AuthView` tidak memanggil Supabase langsung. `HistoryRepository` hanya berurusan dengan persistensi riwayat, sedangkan classifier, preprocessor, OCR, dan heuristics tetap independen dari autentikasi.

Classifier tidak boleh dimuat sebelum autentikasi berhasil. Hal ini mencegah penggunaan model oleh pengguna anonim dan menghindari biaya startup model pada halaman login.

## Konfigurasi dan Secrets

Aplikasi membutuhkan nilai berikut melalui `.streamlit/secrets.toml` saat lokal atau secret manager platform deployment:

- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`
- `APP_URL`, yaitu URL lokal atau deployment yang masuk ke allowlist redirect Supabase

Publishable key aman digunakan bersama JWT pengguna dan RLS. `service_role`, secret key, password database, dan access token management Supabase tidak digunakan oleh aplikasi dan tidak boleh dimasukkan ke repository atau log.

Client autentikasi tidak boleh memakai `st.cache_resource` karena state auth pada client bersifat mutable dan client global dapat mencampur sesi antarpengguna. Setiap sesi Streamlit memiliki instance client serta access token dan refresh token sendiri di `st.session_state`.

## Model Data

### Identitas pengguna

Supabase mengelola email, password hash, status konfirmasi, sesi, dan UUID pengguna pada schema `auth`. Nama lengkap dikirim saat sign up dan disimpan sebagai `user_metadata.full_name`. Metadata tersebut hanya digunakan untuk tampilan dan tidak boleh digunakan dalam policy atau keputusan otorisasi.

### Tabel `public.analysis_history`

| Kolom | Tipe | Aturan |
| --- | --- | --- |
| `id` | `uuid` | Primary key, default UUID acak |
| `user_id` | `uuid` | Wajib, foreign key ke `auth.users(id)`, `ON DELETE CASCADE` |
| `input_text` | `text` | Wajib, teks lengkap input langsung atau hasil OCR |
| `input_source` | `text` | Wajib, hanya `text` atau `image` |
| `prediction_label` | `text` | Wajib, label yang dikembalikan classifier |
| `confidence` | `double precision` | Wajib, antara `0` dan `1` |
| `red_flags` | `jsonb` | Wajib, array string, default `[]` |
| `created_at` | `timestamptz` | Wajib, default waktu database saat insert |

Indeks `(user_id, created_at DESC)` mendukung daftar riwayat terbaru per pengguna. Gambar asli tidak disimpan; untuk input gambar, hanya teks hasil OCR dan `input_source = 'image'` yang disimpan.

## Hak Akses dan RLS

RLS wajib aktif pada `public.analysis_history`. Role `anon` tidak mendapat hak tabel. Role `authenticated` hanya mendapat `SELECT` dan `INSERT` karena perubahan platform Supabase dapat membuat tabel baru tidak otomatis tersedia melalui Data API.

Policy yang diperlukan:

- `SELECT TO authenticated USING ((select auth.uid()) = user_id)`
- `INSERT TO authenticated WITH CHECK ((select auth.uid()) = user_id)`

Tidak ada grant atau policy untuk `UPDATE` dan `DELETE`. Aplikasi selalu mengambil `user_id` dari sesi terverifikasi, bukan dari input form. Walaupun demikian, RLS tetap menolak insert dengan UUID pengguna lain.

Publishable key digunakan untuk seluruh permintaan data. Menggunakan `service_role` dilarang karena key tersebut melewati RLS.

## Alur Pengguna

### Startup dan auth gate

1. `st.set_page_config` dijalankan terlebih dahulu.
2. Aplikasi memvalidasi keberadaan konfigurasi Supabase tanpa mencetak nilainya.
3. Sesi yang sudah ada dipulihkan dan token kedaluwarsa dicoba refresh satu kali.
4. Jika tidak ada sesi valid, hanya halaman autentikasi yang dirender dan classifier tidak dimuat.
5. Jika sesi valid, sidebar menampilkan nama pengguna, menu Analyze/History, dan Logout.

### Sign up dan verifikasi email

1. Pengguna mengisi nama lengkap, email, password, dan konfirmasi password.
2. Aplikasi memangkas nama, menormalisasi email, memvalidasi format email, memastikan nama tidak kosong, dan mewajibkan password minimal delapan karakter.
3. `AuthService` memanggil sign up Supabase dengan `full_name` sebagai user metadata.
4. UI selalu memberi instruksi untuk memeriksa email jika request dapat diterima.
5. Template email mengarahkan token hash sekali pakai ke `APP_URL` yang sudah di-allowlist.
6. Aplikasi memverifikasi token melalui Supabase, membersihkan parameter token dari URL, lalu mengarahkan pengguna ke login atau sesi terverifikasi yang dihasilkan.

### Login dan sesi

1. Pengguna login dengan email dan password.
2. Respons sukses menyimpan access token, refresh token, UUID, email, dan nama tampilan dalam `st.session_state`.
3. Rerun Streamlit menggunakan kembali sesi tersebut dan memasang JWT pada client milik sesi.
4. Token kedaluwarsa di-refresh satu kali. Refresh yang gagal menghapus state auth dan meminta login ulang.
5. Sesi tidak dijanjikan bertahan setelah sesi Streamlit/browser benar-benar berakhir; remember me berada di luar cakupan.

### Lupa dan reset password

1. Pengguna meminta reset dengan email.
2. UI selalu menampilkan pesan generik, terlepas dari apakah email terdaftar, untuk mengurangi account enumeration.
3. Supabase mengirim email recovery dengan token hash sekali pakai.
4. Aplikasi membaca `token_hash` dan tipe `recovery`, memverifikasinya melalui `verify_otp`, lalu langsung membersihkan parameter URL.
5. Setelah sesi recovery valid, pengguna mengisi password baru dan konfirmasinya.
6. Aplikasi memanggil update password Supabase, mengakhiri state recovery, lalu meminta login memakai password baru.

### Analisis dan penyimpanan riwayat

1. Pengguna memilih input teks atau gambar.
2. Flow OCR, preprocessing, classifier, dan heuristics berjalan seperti saat ini.
3. Hasil langsung dirender agar kegagalan database tidak menghilangkan hasil inferensi.
4. `HistoryRepository` menyisipkan teks lengkap, sumber input, label, confidence, red flags, serta UUID pengguna aktif.
5. Jika insert gagal, hasil tetap terlihat dan UI menjelaskan bahwa riwayat belum tersimpan.

### Riwayat

1. Riwayat awal menampilkan 20 item terbaru.
2. Pengguna dapat memuat halaman berikutnya tanpa mengambil seluruh data sekaligus.
3. Setiap item menampilkan waktu, label, confidence, dan cuplikan teks.
4. Detail menampilkan teks lengkap, sumber input, label, confidence, dan seluruh red flags.
5. Empty state dan error state ditampilkan secara eksplisit.

### Logout

1. Aplikasi memanggil sign out Supabase.
2. Semua token dan identitas pengguna dihapus dari session state meskipun request sign out jarak jauh gagal.
3. Aplikasi melakukan rerun dan kembali ke halaman login.

## Template Email dan SMTP

Template Confirm Sign Up dan Reset Password dikustomisasi di dashboard Supabase agar memakai `{{ .TokenHash }}` dan mengarah ke `APP_URL`. Pola token hash dipilih karena fragment URL default tidak tersedia bagi server Streamlit. Token diverifikasi melalui request POST Supabase dan dihapus dari query string segera setelah diproses.

Untuk demo skripsi, SMTP bawaan Supabase hanya dapat mengirim ke alamat yang menjadi anggota tim proyek dan memiliki rate limit serta layanan best-effort. Email demo harus menggunakan alamat yang telah diotorisasi tersebut.

Sebelum deployment publik:

- konfigurasi custom SMTP di dashboard Supabase;
- gunakan domain pengirim yang valid dan atur SPF, DKIM, serta DMARC;
- sesuaikan auth rate limits;
- aktifkan CAPTCHA untuk sign up, login, dan reset password bila endpoint sudah dibuka ke publik;
- pastikan Site URL dan redirect allowlist hanya berisi origin yang dipercaya.

Custom SMTP merupakan perubahan konfigurasi operasional; kode aplikasi dan schema tidak perlu dirombak saat fitur tersebut ditambahkan.

## Validasi dan Penanganan Error

- Nama wajib setelah whitespace dipangkas.
- Email dinormalisasi ke lowercase dan divalidasi sebelum request.
- Password minimal delapan karakter dan harus sama dengan konfirmasi.
- Teks analisis mengikuti batas 1.500 kata yang sudah ada.
- Confidence hanya disimpan jika berada pada rentang `0` sampai `1`.
- Red flags dinormalisasi menjadi array string.
- Pesan login tidak membedakan email tidak terdaftar dan password salah.
- Pesan lupa password tidak mengungkap keberadaan akun.
- Error jaringan memberikan pesan yang dapat dicoba ulang tanpa stack trace atau secrets.
- Error konfigurasi menyebut nama secret yang hilang, bukan nilainya.
- Token invalid/kedaluwarsa tidak diterima dan query parameter sensitif dibersihkan.
- Error pemuatan history tidak memengaruhi halaman Analyze.
- Error penyimpanan history tidak membatalkan hasil inferensi.

## Strategi Pengujian

### Unit test

Gunakan `unittest` dan fake client agar test tidak memerlukan jaringan untuk:

- validasi sign up dan password baru;
- sign up, login, refresh, logout, serta pemetaan error pada `AuthService`;
- pemulihan dan pembersihan session state;
- payload insert serta pagination pada `HistoryRepository`;
- auth gate yang memastikan classifier tidak dimuat sebelum login;
- perilaku ketika penyimpanan history gagal setelah inferensi berhasil.

### Test integrasi Supabase

Gunakan project pengembangan atau local Supabase untuk membuktikan:

- pengguna A dapat insert dan select riwayat miliknya;
- pengguna A tidak dapat membaca riwayat pengguna B;
- pengguna A tidak dapat insert dengan `user_id` pengguna B;
- role anonim tidak dapat select atau insert;
- role authenticated tidak dapat update atau delete;
- foreign key dan constraint confidence/input source bekerja;
- index tersedia;
- Security Advisor tidak melaporkan RLS yang hilang atau policy berbahaya.

### Smoke test UI

Uji sign up, konfirmasi email, login, analisis teks, analisis gambar, penyimpanan history, pagination/detail, reset password, refresh token, invalid token, logout, serta startup dengan secret yang hilang.

Validasi repository mencakup `python -m compileall`, unit test, dan smoke test Streamlit. Test integrasi email memakai alamat anggota tim selama fase demo.

## Kriteria Penerimaan

- Pengguna anonim tidak dapat membuka Analyze atau History dan tidak memicu pemuatan classifier.
- Pengguna dapat sign up dengan nama/email/password, memverifikasi email, login, reset password, dan logout.
- Token recovery tidak tertinggal di URL setelah diproses.
- Setiap analisis berhasil mencoba membuat satu riwayat dengan teks lengkap dan hasil yang benar.
- History hanya menampilkan data pemilik sesi dan memuat maksimal 20 item per permintaan.
- RLS test membuktikan isolasi dua pengguna serta penolakan akses anonim, update, dan delete.
- Tidak ada service role, secret key, token pengguna, atau kredensial lain dalam repository dan log.
- Aplikasi tetap menampilkan hasil inferensi ketika penyimpanan history gagal.
- Dokumentasi menjelaskan konfigurasi demo serta prasyarat custom SMTP untuk publik.

## Referensi Resmi

- [Supabase Auth](https://supabase.com/docs/guides/auth)
- [Password-based Auth](https://supabase.com/docs/guides/auth/passwords)
- [Python Auth reference](https://supabase.com/docs/reference/python/auth-signup)
- [Email templates](https://supabase.com/docs/guides/auth/auth-email-templates)
- [Custom SMTP](https://supabase.com/docs/guides/auth/auth-smtp)
- [Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security)
- [Securing the Data API](https://supabase.com/docs/guides/api/securing-your-api)
- [Production checklist](https://supabase.com/docs/guides/deployment/going-into-prod)
