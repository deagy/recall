package connector

import (
	"testing"
)

// Vectors cross-checked against an independent Python SigV4 implementation
// (HMAC-SHA256 key-derivation chain and canonical request construction).

func TestSigV4_GetVanilla(t *testing.T) {
	headers := map[string]string{
		"host":       "iam.amazonaws.com",
		"x-amz-date": "20150830T123600Z",
	}
	payloadHash := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	creq := buildCanonicalRequest("GET", "/", "", headers, payloadHash)
	wantCREQ := "GET\n/\n\nhost:iam.amazonaws.com\nx-amz-date:20150830T123600Z\nhost;x-amz-date\n" + payloadHash
	if creq != wantCREQ {
		t.Errorf("canonical request mismatch:\ngot  %q\nwant %q", creq, wantCREQ)
	}
	sts := stringToSign(creq, "20150830T123600Z", "20150830/us-east-1/iam/aws4_request")
	const wantSTSHash = "80f78ceb057168a133b26331ac35c063ddceb011c481d9280681ad80e805d74c"
	wantSTS := "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/iam/aws4_request\n" + wantSTSHash
	if sts != wantSTS {
		t.Errorf("string-to-sign mismatch:\ngot  %q\nwant %q", sts, wantSTS)
	}
	got := sigV4Signature(sts, "20150830T123600Z", "20150830/us-east-1/iam/aws4_request", "us-east-1", "iam", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
	const wantSig = "ee1f902035fbdd9783600ce7e823bb3dc7b6f9a7f13e867e333aa999530ff18a"
	if got != wantSig {
		t.Errorf("signature: got %s want %s", got, wantSig)
	}
}

func TestSigV4_S3List(t *testing.T) {
	headers := map[string]string{
		"host":                 "mybucket.s3.us-west-2.amazonaws.com",
		"x-amz-content-sha256": "UNSIGNED-PAYLOAD",
		"x-amz-date":           "20260818T120000Z",
	}
	creq := buildCanonicalRequest("GET", "/", "list-type=2&prefix=docs%2Freports%2F2026", headers, "UNSIGNED-PAYLOAD")
	sts := stringToSign(creq, "20260818T120000Z", "20260818/us-west-2/s3/aws4_request")
	got := sigV4Signature(sts, "20260818T120000Z", "20260818/us-west-2/s3/aws4_request", "us-west-2", "s3", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
	const wantSig = "fd11bcb164d46a5d33759353eaf4bc2384c258d622bedb25598985dc784f4b1a"
	if got != wantSig {
		t.Errorf("signature: got %s want %s", got, wantSig)
	}
}

func TestSigV4_PathStyleKey(t *testing.T) {
	headers := map[string]string{
		"host":                 "s3.us-west-2.amazonaws.com",
		"x-amz-content-sha256": "UNSIGNED-PAYLOAD",
		"x-amz-date":           "20260818T120000Z",
	}
	// The canonical URI escapes decoded segments; '/' stays a separator.
	if got := canonicalURI("/mybucket/docs/report 1.pdf"); got != "/mybucket/docs/report%201.pdf" {
		t.Errorf("canonicalURI: got %q", got)
	}
	creq := buildCanonicalRequest("GET", "/mybucket/docs/report%201.pdf", "", headers, "UNSIGNED-PAYLOAD")
	sts := stringToSign(creq, "20260818T120000Z", "20260818/us-west-2/s3/aws4_request")
	got := sigV4Signature(sts, "20260818T120000Z", "20260818/us-west-2/s3/aws4_request", "us-west-2", "s3", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
	const wantSig = "37204396fb15f822ced118bc8894fdae9cde1fa52f6326f6258e16d834aaf490"
	if got != wantSig {
		t.Errorf("signature: got %s want %s", got, wantSig)
	}
}

func TestCanonicalQueryString_Sorting(t *testing.T) {
	// Keys sorted alphabetically; values URI-encoded (space -> %20, '/' -> %2F).
	got := canonicalQueryString(map[string][]string{
		"prefix":    {"docs/reports 2026"},
		"list-type": {"2"},
		"max-keys":  {"100"},
	})
	want := "list-type=2&max-keys=100&prefix=docs%2Freports%202026"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
	if got := canonicalQueryString(nil); got != "" {
		t.Errorf("empty query: got %q", got)
	}
}

func TestParseS3Ref(t *testing.T) {
	bucket, prefix, err := parseS3Ref("mybucket")
	if err != nil || bucket != "mybucket" || prefix != "" {
		t.Errorf("plain: %v %q %q", err, bucket, prefix)
	}
	bucket, prefix, err = parseS3Ref("s3://b/p/a")
	if err != nil || bucket != "b" || prefix != "p/a" {
		t.Errorf("uri: %v %q %q", err, bucket, prefix)
	}
	if _, _, err := parseS3Ref(""); err == nil {
		t.Error("expected error for empty ref")
	}
}
