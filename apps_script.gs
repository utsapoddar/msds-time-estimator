/**
 * MSDS time-estimator form router.
 *
 * Setup (one-time):
 * 1. In the Google Sheet linked to your form: Extensions → Apps Script.
 * 2. Paste this file's contents into Code.gs. Save.
 * 3. Triggers (clock icon) → Add trigger: function=onFormSubmit,
 *    event source=From spreadsheet, event type=On form submit.
 * 4. Replace OWNER_EMAIL below with your email.
 * 5. Run onFormSubmit once manually to authorize Gmail + Sheets scopes.
 *
 * Behavior:
 * - Adds a "review_needed" column (auto/review) to the Form Responses sheet on each submit.
 * - Emails OWNER_EMAIL when a submission is flagged.
 * - Does NOT duplicate rows across tabs; form_intake.py splits locally.
 */

const OWNER_EMAIL = 'utsapoddar@gmail.com';
const MIN_DAYS = 30;
const MAX_DAYS = 90;
const MIN_HOURS_PER_WEEK = 2;
const MAX_HOURS_PER_WEEK = 60;

// Field names must exactly match the question titles in your form.
const F_START = 'Start date';
const F_END   = 'End date (last assignment submitted)';
const F_HPW   = 'Hours per week available during the course';
const F_NOTES = 'Notes or anything unusual about your experience';
const F_COURSE = 'Which course?';
const FLAG_COL_HEADER = 'review_needed';

function onFormSubmit(e) {
  const sheet = e.range.getSheet();
  const row = e.range.getRow();
  const nv = e.namedValues || {};

  const startStr = (nv[F_START] || [''])[0];
  const endStr   = (nv[F_END]   || [''])[0];
  const days = (startStr && endStr)
    ? Math.round((new Date(endStr) - new Date(startStr)) / 86400000)
    : NaN;
  const hpw  = parseFloat((nv[F_HPW]  || [''])[0]);
  const notes = ((nv[F_NOTES] || [''])[0] || '').trim();
  const course = (nv[F_COURSE] || [''])[0];

  const reasons = [];
  if (isFinite(days) && (days < MIN_DAYS || days > MAX_DAYS)) {
    reasons.push(`days=${days} outside [${MIN_DAYS},${MAX_DAYS}]`);
  }
  if (isFinite(hpw) && (hpw < MIN_HOURS_PER_WEEK || hpw > MAX_HOURS_PER_WEEK)) {
    reasons.push(`hours/week=${hpw} outside [${MIN_HOURS_PER_WEEK},${MAX_HOURS_PER_WEEK}]`);
  }
  if (notes.length > 0) reasons.push('notes field populated');

  const flag = reasons.length > 0 ? 'review' : 'auto';
  ensureFlagColumn_(sheet);
  const flagCol = findColumn_(sheet, FLAG_COL_HEADER);
  sheet.getRange(row, flagCol).setValue(flag);

  if (flag === 'review') {
    const subject = `MSDS estimator: ${course} submission flagged for review`;
    const body =
      `Row ${row} flagged.\n\n` +
      `Reasons:\n  - ${reasons.join('\n  - ')}\n\n` +
      `Course: ${course}\nDays: ${days}\nHours/week: ${hpw}\n` +
      `Notes: ${notes || '(none)'}\n\n` +
      `Sheet: ${sheet.getParent().getUrl()}`;
    MailApp.sendEmail(OWNER_EMAIL, subject, body);
  }
}

function ensureFlagColumn_(sheet) {
  const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  if (!headers.includes(FLAG_COL_HEADER)) {
    sheet.getRange(1, sheet.getLastColumn() + 1).setValue(FLAG_COL_HEADER);
  }
}

function findColumn_(sheet, header) {
  const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  return headers.indexOf(header) + 1;
}
